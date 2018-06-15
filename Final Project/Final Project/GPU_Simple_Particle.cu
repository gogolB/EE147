#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPU_Simple_Particle.cuh"

#include "GPU_Setup.h"

#include "Helper.h"
#include "Texture.h"

#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>

__global__ void GPU_simulate(GPU_Particle* particles, int maxParticles, float delta, glm::vec3 CameraPosition, GLfloat* pos_buffer, GLubyte * color_buffer)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;

	if (i > maxParticles - 1)
		return;

	GPU_Particle& p = particles[i];
	
	if (p.life > 0.0f) 
	{

		// Decrease life
		p.life -= delta;
		if (p.life > 0.0f) 
		{

			// Simulate simple physics : gravity only, no collisions
			p.speed += glm::vec3(0.0f, -9.81f, 0.0f) * (float)delta * 0.5f;
			p.pos += p.speed * (float)delta;
			p.cameradistance = glm::length2(p.pos - CameraPosition);

			// Fill the GPU buffer
			
			pos_buffer[4 * i + 0] = p.pos.x;
			pos_buffer[4 * i + 1] = p.pos.y;
			pos_buffer[4 * i + 2] = p.pos.z;

			pos_buffer[4 * i + 3] = 0.1;

			color_buffer[4 * i + 0] = 128;
			color_buffer[4 * i + 1] = 128;
			color_buffer[4 * i + 2] = 128;
			color_buffer[4 * i + 3] = 128;
		}
		else
		{
			// Particles that just died will be put at the end of the buffer in SortParticles();
			p.cameradistance = -1.0f;
		}
	}
}
__host__ void initParticles(GPU_Particle* particles, int maxP)
{

	for (int i = 0; i < maxP; i++)
	{
	
		GPU_Particle& p = particles[i];
		p.life = 5.0f + (rand() % 10); // This particle will live 5 seconds.
		p.pos = glm::vec3(0, 0, -20.0f);

		float spread = 1.5f;
		glm::vec3 maindir = glm::vec3(0.0f, 10.0f, 0.0f);
		// Very bad way to generate a random direction; 
		// See for instance http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution instead,
		// combined with some user-controlled parameters (main direction, spread, etc)
		glm::vec3 randomdir = glm::vec3(
			(rand() % 2000 - 1000.0f) / 1000.0f,
			(rand() % 2000 - 1000.0f) / 1000.0f,
			(rand() % 2000 - 1000.0f) / 1000.0f
		);

		p.speed = maindir + randomdir*spread;
	}
}


#pragma region helper

void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags, unsigned int size)
{
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_STREAM_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

	// unregister this buffer object with CUDA
	cudaGraphicsUnregisterResource(vbo_res);

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

int FindUnusedParticle(int& LastUsedParticle, int MaxParticles, GPU_Particle * ParticlesContainer) {

	for (int i = LastUsedParticle; i < MaxParticles; i++) {
		if (ParticlesContainer[i].life < 0) {
			LastUsedParticle = i;
			return i;
		}
	}

	for (int i = 0; i < LastUsedParticle; i++) {
		if (ParticlesContainer[i].life < 0) {
			LastUsedParticle = i;
			return i;
		}
	}

	return -1; // All particles are taken, override the first one
}

#pragma endregion


__host__ int GPU_Simple_Particle::init()
{
	printf("Initializing particles on GPU...\n");
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	programID = LoadShaders("./Shaders/Particle.v", "./Shaders/Particle.f");

	CameraRight_worldspace_ID = glGetUniformLocation(programID, "CameraRight_worldspace");
	CameraUp_worldspace_ID = glGetUniformLocation(programID, "CameraUp_worldspace");
	ViewProjMatrixID = glGetUniformLocation(programID, "VP");

	TextureID = glGetUniformLocation(programID, "myTextureSampler");

	g_particule_position_size_data = new GLfloat[MaxParticles * 4];
	g_particule_color_data = new GLubyte[MaxParticles * 4];

	Texture = loadDDS("./Shaders/particle.DDS");

	// The VBO containing the 4 vertices of the particles.
	// Thanks to instancing, they will be shared by all particles.
	static const GLfloat g_vertex_buffer_data[] = {
		-0.5f, -0.5f, 0.0f,
		0.5f, -0.5f, 0.0f,
		-0.5f,  0.5f, 0.0f,
		0.5f,  0.5f, 0.0f,
	};
	glGenBuffers(1, &billboard_vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	// The VBO containing the positions and sizes of the particles
	createVBO(&particles_position_buffer, &cuda_pos_resource, cudaGraphicsMapFlagsWriteDiscard, MaxParticles * 4 * sizeof(GLfloat));
	/*
	glGenBuffers(1, &particles_position_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	// Initialize with empty (NULL) buffer : it will be updated later, each frame.
	glBufferData(GL_ARRAY_BUFFER, MaxParticles * 4 * sizeof(GLfloat), NULL, GL_DYNAMIC_COPY);
	// Notify Cuda that it is a buffer we can edit on GPU.
	cudaGLRegisterBufferObject(particles_position_buffer);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	*/

	// The VBO containing the colors of the particles
	createVBO(&particles_color_buffer, &cuda_color_resource, cudaGraphicsMapFlagsWriteDiscard, MaxParticles * 4 * sizeof(GLubyte));
	/*
	glGenBuffers(1, &particles_color_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
	// Initialize with empty (NULL) buffer : it will be updated later, each frame.
	glBufferData(GL_ARRAY_BUFFER, MaxParticles * 4 * sizeof(GLubyte), NULL, GL_DYNAMIC_COPY);
	// Notify Cuda that it is a buffer we can edit on GPU.
	cudaGLRegisterBufferObject(particles_color_buffer);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	*/
	// Create the GPU particle array.
	cudaMallocManaged(&g_particule_position_size_data, MaxParticles * sizeof(GLfloat) * 4);
	cudaMallocManaged(&g_particule_color_data, MaxParticles * sizeof(GLfloat) * 4);
	cudaMallocManaged(&UArray, MaxParticles * sizeof(GPU_Particle));
	// Init it.
	initParticles (UArray, MaxParticles);
	// Sync the particles
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

//	GPU_Particle p;
	/*
	for (int i = 0; i < MaxParticles; i++)
	{
		GPU_Particle p = UArray[i];
		printf("%d[%f,%f,%f,%f]\n", i, p.pos.x, p.pos.y, p.pos.z, p.life);
		printf("-[%f,%f,%f]\n", p.speed.x, p.speed.y, p.speed.z);
	}*/
	return 0;
}


__host__ void GPU_Simple_Particle::cleanup()
{
	// Unregester before VBO cleanup
	deleteVBO(&particles_position_buffer, cuda_pos_resource);
	deleteVBO(&particles_color_buffer, cuda_color_resource);

	glDeleteBuffers(1, &billboard_vertex_buffer);
	glDeleteProgram(programID);
	glDeleteTextures(1, &Texture);
	glDeleteVertexArrays(1, &VertexArrayID);

	// clean up particle array.
	cudaFree(UArray);
	cudaFree(g_particule_color_data);
	cudaFree(g_particule_position_size_data);
}

__host__ void GPU_Simple_Particle::simulate(float delta, glm::vec3 cameraPos)
{
	/*
	GLfloat * posPointer;
	GLubyte * colorPointer;
	// Map the buffer to CUDA
	cudaGraphicsMapResources(1, &cuda_pos_resource, 0);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void **)&posPointer, &num_bytes, cuda_pos_resource);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	cudaGraphicsMapResources(1, &cuda_color_resource, 0);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	cudaGraphicsResourceGetMappedPointer((void **)&colorPointer, &num_bytes, cuda_color_resource);
	err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
		*/

	dim3 DimBlock(512, 1, 1);
	dim3 DimGrid((MaxParticles - 1) / 512 + 1, 1, 1);

	// Call the simulate function
	GPU_simulate <<<DimGrid, DimBlock >>> (UArray, MaxParticles, delta, cameraPos, g_particule_position_size_data, g_particule_color_data);
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));
	int lastParticle = 0;
	while (lastParticle >= 0)
	{
		int i = FindUnusedParticle(lastParticle, MaxParticles, UArray);
		if (i == -1)
			break;

		GPU_Particle& p = UArray[i];
		p.life = 5.0f + (rand() % 10); // This particle will live 5 seconds.
		p.pos = glm::vec3(0, 0, -20.0f);

		float spread = 1.5f;
		glm::vec3 maindir = glm::vec3(0.0f, 10.0f, 0.0f);
		// Very bad way to generate a random direction; 
		// See for instance http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution instead,
		// combined with some user-controlled parameters (main direction, spread, etc)
		glm::vec3 randomdir = glm::vec3(
			(rand() % 2000 - 1000.0f) / 1000.0f,
			(rand() % 2000 - 1000.0f) / 1000.0f,
			(rand() % 2000 - 1000.0f) / 1000.0f
		);

		p.speed = maindir + randomdir*spread;
	}

	//cudaGraphicsUnmapResources(1, &cuda_pos_resource,0);
	//cudaGraphicsUnmapResources(1, &cuda_color_resource,0);
}

__host__ void GPU_Simple_Particle::render(float delta, glm::vec3 CameraPosition, glm::mat4 ViewProjectionMatrix, glm::mat4 ViewMatrix)
{
	// Update the buffers that OpenGL uses for rendering.
	// There are much more sophisticated means to stream data from the CPU to the GPU, 
	// but this is outside the scope of this tutorial.
	// http://www.opengl.org/wiki/Buffer_Object_Streaming
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	glBufferData(GL_ARRAY_BUFFER, MaxParticles * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW); // Buffer orphaning, a common way to improve streaming perf. See above link for details.
	glBufferSubData(GL_ARRAY_BUFFER, 0, MaxParticles * sizeof(GLfloat) * 4, g_particule_position_size_data);

	glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
	glBufferData(GL_ARRAY_BUFFER, MaxParticles * 4 * sizeof(GLubyte), NULL, GL_STREAM_DRAW); // Buffer orphaning, a common way to improve streaming perf. See above link for details.
	glBufferSubData(GL_ARRAY_BUFFER, 0, MaxParticles * sizeof(GLubyte) * 4, g_particule_color_data);


	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Use our shader
	glUseProgram(programID);

	// Bind our texture in Texture Unit 0
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, Texture);
	// Set our "myTextureSampler" sampler to use Texture Unit 0
	glUniform1i(TextureID, 0);

	// Same as the billboards tutorial
	glUniform3f(CameraRight_worldspace_ID, ViewMatrix[0][0], ViewMatrix[1][0], ViewMatrix[2][0]);
	glUniform3f(CameraUp_worldspace_ID, ViewMatrix[0][1], ViewMatrix[1][1], ViewMatrix[2][1]);

	glUniformMatrix4fv(ViewProjMatrixID, 1, GL_FALSE, &ViewProjectionMatrix[0][0]);

	// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
	glVertexAttribPointer(
		0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);

	// 2nd attribute buffer : positions of particles' centers
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	glVertexAttribPointer(
		1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
		4,                                // size : x + y + z + size => 4
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
	);

	// 3rd attribute buffer : particles' colors
	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
	glVertexAttribPointer(
		2,                                // attribute. No particular reason for 1, but must match the layout in the shader.
		4,                                // size : r + g + b + a => 4
		GL_UNSIGNED_BYTE,                 // type
		GL_TRUE,                          // normalized?    *** YES, this means that the unsigned char[4] will be accessible with a vec4 (floats) in the shader ***
		0,                                // stride
		(void*)0                          // array buffer offset
	);

	// These functions are specific to glDrawArrays*Instanced*.
	// The first parameter is the attribute buffer we're talking about.
	// The second parameter is the "rate at which generic vertex attributes advance when rendering multiple instances"
	// http://www.opengl.org/sdk/docs/man/xhtml/glVertexAttribDivisor.xml
	glVertexAttribDivisor(0, 0); // particles vertices : always reuse the same 4 vertices -> 0
	glVertexAttribDivisor(1, 1); // positions : one per quad (its center)                 -> 1
	glVertexAttribDivisor(2, 1); // color : one per quad                                  -> 1

								 // Draw the particules !
								 // This draws many times a small triangle_strip (which looks like a quad).
								 // This is equivalent to :
								 // for(i in ParticlesCount) : glDrawArrays(GL_TRIANGLE_STRIP, 0, 4), 
								 // but faster.
	glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, MaxParticles);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
}
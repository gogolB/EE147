
#include "GPU_Particle.cuh"
#include <stdio.h>

#include "Helper.h"
#include "Texture.h"

int GPU_Particle_Sim::init()
{

	printf("Starting init...\n");

	unsigned int n = 3 * size_x*size_y*size_z;
	unsigned int n_p = 7* MaxParticles; //1 million points defined by 7 flos each

	cudaMalloc((void **)&u, n * sizeof(float));
	cudaMallocManaged(&p, n_p * sizeof(float));

	force_h = (float*)malloc(3 * sizeof(float));
	pos_h = (float*)malloc(3 * sizeof(float));
	grav_h = (float*)malloc(3 * sizeof(float));

	force_h[0] = 0.5; grav_h[0] = 0;   pos_h[0] = 1.25;
	force_h[1] = 0.5; grav_h[1] = -9.8; pos_h[1] = 1.25;
	force_h[2] = 0.5; grav_h[2] = 0;   pos_h[2] = 1.25;

	cudaMalloc((void**)&force_d, 3 * sizeof(float));
	cudaMalloc((void**)&grav_d, 3 * sizeof(float));
	cudaMalloc((void**)&pos_d, 3 * sizeof(float));
	cudaError cuda_ret = cudaDeviceSynchronize();
	if (cuda_ret != cudaSuccess) {
		printf("Error: failed to allocated device variables\n\tThrew: %s\n", cudaGetErrorString(cuda_ret)); fflush(stdout);
		return -1;
	}

	cuda_ret = cudaMemcpy(force_d, force_h, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(grav_d, grav_h, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(pos_d, pos_h, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cuda_ret = cudaDeviceSynchronize();
	if (cuda_ret != cudaSuccess) {
		printf("Error: Memcpy of initial values failed\n\tThrew: %s\n", cudaGetErrorString(cuda_ret)); fflush(stdout);
		return -1;
	}


	dim3 blockDim_1D(1000, 1, 1);
	dim3 gridDim_1D((n - 1) / 1000 + 1, 1, 1);

	zeroVector <<<gridDim_1D, blockDim_1D >> >(u, n);
	cuda_ret = cudaDeviceSynchronize();
	if (cuda_ret != cudaSuccess) {
		printf("Error: Failed to initialize velocity field\n\tThrew: %s\n", cudaGetErrorString(cuda_ret)); fflush(stdout);
		return -1;
	}

	initPoints(p, MaxParticles, size_x, size_y, size_z);

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
	glGenBuffers(1, &particles_position_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	// Initialize with empty (NULL) buffer : it will be updated later, each frame.
	glBufferData(GL_ARRAY_BUFFER, MaxParticles * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW);

	// The VBO containing the colors of the particles
	glGenBuffers(1, &particles_color_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
	// Initialize with empty (NULL) buffer : it will be updated later, each frame.
	glBufferData(GL_ARRAY_BUFFER, MaxParticles * 4 * sizeof(GLubyte), NULL, GL_STREAM_DRAW);

	return 0;
}

void GPU_Particle_Sim::simulate(float delta, glm::vec3 cameraPos)
{
		globalForce(u, grav_d, size_x, size_y, size_z);
		cudaDeviceSynchronize();
		//if (i > 1 / DELTA_T && i < 3 / DELTA_T) {
		//	localForce(u, force_d, pos_d, 0.25, size_x, size_y, size_z);
		//}
		//cudaDeviceSynchronize();

		advection(u, size_x, size_y, size_z);

		diffusion(u, size_x, size_y, size_z);

		project(u, size_x, size_y, size_z);

		//cudaDeviceSynchronize();

		updatePoints(u, p, MaxParticles, size_x, size_y, size_z);

		cudaDeviceSynchronize();

		ParticlesCount = 0;
		for (int i = 0; i < MaxParticles; i++) {

			//ParticlesContainer[i].pos += glm::vec3(0.0f,10.0f, 0.0f) * (float)delta;

			// Fill the GPU buffer
			g_particule_position_size_data[4 * i + 0] = p[7 * i + 0];
			g_particule_position_size_data[4 * i + 1] = p[7 * i + 1];
			g_particule_position_size_data[4 * i + 2] = p[7 * i + 2];

			// particle size.
			g_particule_position_size_data[4 * i + 3] = 0.1;

			g_particule_color_data[4 * i + 0] = 128;
			g_particule_color_data[4 * i + 1] = 128;
			g_particule_color_data[4 * i + 2] = 128;
			g_particule_color_data[4 * i + 3] = 128;


			ParticlesCount++;

		}
}

void GPU_Particle_Sim::render(float delta, glm::vec3 CameraPosition, glm::mat4 ViewProjectionMatrix, glm::mat4 ViewMatrix)
{
	printf("[%f,%f,%f,%f,%f,%f,%f]\n", p[0], p[1], p[2], p[3], p[4], p[5], p[6] );
	// Update the buffers that OpenGL uses for rendering.
	// There are much more sophisticated means to stream data from the CPU to the GPU, 
	// but this is outside the scope of this tutorial.
	// http://www.opengl.org/wiki/Buffer_Object_Streaming
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	glBufferData(GL_ARRAY_BUFFER, MaxParticles * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW); // Buffer orphaning, a common way to improve streaming perf. See above link for details.
	glBufferSubData(GL_ARRAY_BUFFER, 0, ParticlesCount * sizeof(GLfloat) * 4, g_particule_position_size_data);

	glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
	glBufferData(GL_ARRAY_BUFFER, MaxParticles * 4 * sizeof(GLubyte), NULL, GL_STREAM_DRAW); // Buffer orphaning, a common way to improve streaming perf. See above link for details.
	glBufferSubData(GL_ARRAY_BUFFER, 0, ParticlesCount * sizeof(GLubyte) * 4, g_particule_color_data);


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
	glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, ParticlesCount);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
}

void GPU_Particle_Sim::cleanup()
{
	delete[] g_particule_position_size_data;
	delete[] g_particule_color_data;

	// Cleanup VBO and shader
	glDeleteBuffers(1, &particles_color_buffer);
	glDeleteBuffers(1, &particles_position_buffer);
	glDeleteBuffers(1, &billboard_vertex_buffer);
	glDeleteProgram(programID);
	glDeleteTextures(1, &Texture);
	glDeleteVertexArrays(1, &VertexArrayID);

	cudaFree(u);
	cudaFree(force_d);
	cudaFree(pos_d);
	cudaFree(grav_d);
	cudaFree(p);

	free(force_h);
	free(pos_h);
	free(grav_h);
}
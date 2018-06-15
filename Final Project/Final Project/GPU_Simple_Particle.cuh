#pragma once
// Code based on:
// http://www.opengl-tutorial.org/
// https://learnopengl.com/

#define GLM_ENABLE_EXPERIMENTAL

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Include GLEW. Always include it before gl.h and glfw3.h, since it's a bit magic.
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
using namespace glm;

typedef struct GPU_Particle {
	glm::vec3 pos, speed;

	float life = -1; // Remaining life of the particle. if <0 : dead and unused.
	float cameradistance; // *Squared* distance to the camera. if dead : -1.0f

	bool operator<(const GPU_Particle& that) const {
		// Sort in reverse order : far particles drawn first.
		return this->cameradistance > that.cameradistance;
	}
} ;


class GPU_Simple_Particle 
{
	GLuint VertexArrayID;
	GLuint programID;

	GLuint CameraRight_worldspace_ID;
	GLuint CameraUp_worldspace_ID;
	GLuint ViewProjMatrixID;

	GLuint TextureID;

	GLfloat* g_particule_position_size_data;
	GLubyte* g_particule_color_data;
	GLuint Texture;

	GLuint billboard_vertex_buffer;

	GLuint particles_position_buffer;

	GLuint particles_color_buffer;

	static const int MaxParticles = 100000;
	int ParticlesCount = 0;
	int LastUsedParticle = 0;

	GPU_Particle * UArray;

	struct cudaGraphicsResource *cuda_pos_resource;
	struct cudaGraphicsResource *cuda_color_resource;


public:
	__host__ int init();

	__host__ void simulate(float delta, glm::vec3 cameraPos);

	__host__ void render(float delta, glm::vec3 CameraPosition, glm::mat4 ViewProjectionMatrix, glm::mat4 ViewMatrix);

	__host__ void cleanup();
};
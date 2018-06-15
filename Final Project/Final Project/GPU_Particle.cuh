#pragma once
// Code based on:
// http://www.opengl-tutorial.org/
// https://learnopengl.com/

#define GLM_ENABLE_EXPERIMENTAL

// Include GLEW. Always include it before gl.h and glfw3.h, since it's a bit magic.
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/norm.hpp>
using namespace glm;

#include <curand.h>
#include <curand_kernel.h>

#define DELTA_T 0.01667
#define LATTICE_SIZE 0.01

#include "advection/advection.h"
#include "diffusion/diffusion.h"
#include "force/force.h"
#include "projection/project.h"
#include "points/points.h"

#include "util.h"


class GPU_Particle_Sim {

	unsigned int size_x = 200;
	unsigned int size_y = 200;
	unsigned int size_z = 200;

	float *force_h, *grav_h, *pos_h;
	float *force_d, *grav_d, *pos_d;
	float *u, *p;

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

public:
	int init();

	void simulate(float delta, glm::vec3 cameraPos);

	void render(float delta, glm::vec3 CameraPosition, glm::mat4 ViewProjectionMatrix, glm::mat4 ViewMatrix);

	void cleanup();
};
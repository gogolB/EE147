#pragma once

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

#include<vector>
#include<algorithm>


#include "Texture.h"

struct Particle {
	glm::vec3 pos, speed;
	unsigned char r, g, b, a; // Color
	float size, angle, weight;
	float life; // Remaining life of the particle. if <0 : dead and unused.
	float cameradistance; // *Squared* distance to the camera. if dead : -1.0f

	bool operator<(const Particle& that) const {
		// Sort in reverse order : far particles drawn first.
		return this->cameradistance > that.cameradistance;
	}
};

class CPU_Particle
{
private:
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
	Particle ParticlesContainer[MaxParticles];
	int LastUsedParticle = 0;
public:
	CPU_Particle();
	~CPU_Particle();


	void init();
	int FindUnusedParticle();
	void SortParticles();
	void simulate(float delta, glm::vec3 CameraPosition);
	void render(float delta, glm::vec3 CameraPosition, glm::mat4 ViewProjectionMatrix, glm::mat4 ViewMatrix);

	void cleanup();
};


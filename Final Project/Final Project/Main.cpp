#include <stdio.h>
#include <stdlib.h>

#define GLM_ENABLE_EXPERIMENTAL
#define LATTICE_SIZE 0.01
#define DELTA_T 0.0167
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

GLFWwindow* window;

#include "Window_Control.h"
#include "Helper.h"
#include "Controls.h"
#include "Texture.h"
#include "CPU_Particle.h"
#include "GPU_Setup.h"

#ifdef GPU
#include "GPU_Particle.cuh"
#endif
#ifdef GPU_SIMPLE
#include "GPU_Simple_Particle.cuh"
#endif

int main()
{
	// Open a window and create its OpenGL context
	window = initAndCreateWindow();

	// Failed to create window. Close out and clean up.
	if (window == NULL)
		return -1;

	// TODO: Get GPU Code working here somehow.
	initOGL_CUDA();

#if !defined(GPU) && !defined(GPU_SIMPLE)
	printf("Using CPU Particles...\n");
#endif // !GPU
	CPU_Particle* cpu_particle = new CPU_Particle();
	cpu_particle->init();
	
#ifdef GPU_SIMPLE
	printf("USING GPU Simple Particles...\n");
	GPU_Simple_Particle * gpu_particle = new GPU_Simple_Particle();
	if (gpu_particle->init() < 0)
		return -1;
#endif

#ifdef GPU
	printf("USING GPU Particles...\n");
	GPU_Particle_Sim * gpu_particle = new GPU_Particle_Sim();
	if (gpu_particle->init() < 0)
		return -1;
#endif

	printf("Window Created, entering the render loop.\n");
	double lastTime = glfwGetTime();
	// Main render loop...
	do
	{

		// Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		double currentTime = glfwGetTime();
		double delta = currentTime - lastTime;
		lastTime = currentTime;

		printf("Frames per second: %f\n", 1.0 / delta);

		computeMatricesFromInputs();
		glm::mat4 ProjectionMatrix = getProjectionMatrix();
		glm::mat4 ViewMatrix = getViewMatrix();

		glm::vec3 CameraPosition(glm::inverse(ViewMatrix)[3]);

		glm::mat4 ViewProjectionMatrix = ProjectionMatrix * ViewMatrix;

#if  defined(GPU) || defined(GPU_SIMPLE)	
		printf("Sim step\n");
		gpu_particle->simulate(delta, CameraPosition);
		gpu_particle->render(delta, CameraPosition, ViewProjectionMatrix, ViewMatrix);
#else
		cpu_particle->simulate(delta, CameraPosition);
		cpu_particle->render(delta, CameraPosition, ViewProjectionMatrix, ViewMatrix);
#endif

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

		// Check if the ESC key was pressed or the window was closed
	} while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

	printf("Exiting render loop... cleaning up now\n");
	cpu_particle->cleanup();
#if defined(GPU) || defined(GPU_SIMPLE)
	gpu_particle->cleanup();
#endif
	// Close OpenGL window and terminate GLFW
	glfwTerminate();
}
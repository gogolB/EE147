#include <stdio.h>
#include <stdlib.h>

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

GLFWwindow* window;

#include "Window_Control.h"
#include "Helper.h"
#include "Controls.h"
#include "Texture.h"
#include "CPU_Particle.h"
#include "GPU_Setup.h"

int main()
{
	// Open a window and create its OpenGL context
	window = initAndCreateWindow();

	// Failed to create window. Close out and clean up.
	if (window == NULL)
		return -1;

	// TODO: Get GPU Code working here somehow.
	initOGL_CUDA();

	CPU_Particle* cpu_particle = new CPU_Particle();
	cpu_particle->init();



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


		computeMatricesFromInputs();
		glm::mat4 ProjectionMatrix = getProjectionMatrix();
		glm::mat4 ViewMatrix = getViewMatrix();

		// We will need the camera's position in order to sort the particles
		// w.r.t the camera's distance.
		// There should be a getCameraPosition() function in common/controls.cpp, 
		// but this works too.
		glm::vec3 CameraPosition(glm::inverse(ViewMatrix)[3]);

		glm::mat4 ViewProjectionMatrix = ProjectionMatrix * ViewMatrix;


		cpu_particle->render(delta, CameraPosition, ViewProjectionMatrix, ViewMatrix);


		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();

		// Check if the ESC key was pressed or the window was closed
	} while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

	printf("Exiting render loop... cleaning up now\n");
	cpu_particle->cleanup();
	// Close OpenGL window and terminate GLFW
	glfwTerminate();
}
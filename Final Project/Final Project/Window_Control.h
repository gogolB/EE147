#pragma once
// Code based on:
// http://www.opengl-tutorial.org/
// https://learnopengl.com/

// Include GLEW. Always include it before gl.h and glfw3.h, since it's a bit magic.
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>

// Include GLM
#include <glm/glm.hpp>
using namespace glm;

GLFWwindow* initAndCreateWindow();
#pragma once
// Code based on:
// http://www.opengl-tutorial.org/
// https://learnopengl.com/

#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include<glm\glm.hpp>

// Load a .BMP file using our custom loader
GLuint loadBMP_custom(const char * imagepath);

// Load a .DDS file using GLFW's own loader
GLuint loadDDS(const char * imagepath);
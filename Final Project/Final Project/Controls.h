#pragma once
// Code based on:
// http://www.opengl-tutorial.org/
// https://learnopengl.com/

#include<glm\glm.hpp>

void computeMatricesFromInputs();
glm::mat4 getViewMatrix();
glm::mat4 getProjectionMatrix();
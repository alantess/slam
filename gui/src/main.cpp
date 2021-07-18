#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

int main() {
  glfwInit();

  // Describes the OPENGL VERSION
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
  // Tells GLFW we are using core profile
  // // Means we only have the modern functions
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Creates the window
  GLFWwindow *window = glfwCreateWindow(800, 800, "3DPC", NULL, NULL);

  if (window == NULL) {
    std::cout << "Failed to create window";
    glfwTerminate();
    return -1;
  }
  // Sets the window the the current context
  glfwMakeContextCurrent(window);
  // Loads glad
  gladLoadGL(); 
  // sets the viewport
  glViewport(0,0,800,800);
  // Sets a colors and loads the it in by swapping 
  glClearColor(0.07f, 0.13f, 0.17f,1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  glfwSwapBuffers(window);

  // Track all the events within the window I.E. updates it.
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
  }

  glfwDestroyWindow(window);

  glfwTerminate();
  return 0;
}

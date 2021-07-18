#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <GLFW/glfw3.h>
#include <iostream>

int main() {
  glfwInit();

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window = glfwCreateWindow(800, 800, "3DPC", NULL, NULL);

  if (window == NULL) {
    std::cout << "Failed to create window";
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
  }

  glfwDestroyWindow(window);

  glfwTerminate();
  return 0;
}

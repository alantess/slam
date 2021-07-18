cmake_minimum_required(VERSION 3.1)
project(slam)


set(OpenGL_GL_PREFERENCE LEGACY)

find_package(OpenGL REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp
  include/glad/glad.h
  src/glad.c

  )
target_include_directories(slam PRIVATE include)



target_link_libraries(${PROJECT_NAME} OpenGL::GL glfw ${CMAKE_DL_LIBS})

set_property(TARGET slam PROPERTY CXX_STANDARD 20)

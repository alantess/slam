cmake_minimum_required(VERSION 3.1)
project(slam)


set(OpenGL_GL_PREFERENCE LEGACY)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)



target_link_libraries(${PROJECT_NAME} OpenGL::GL GLUT::GLUT glfw)

set_property(TARGET slam PROPERTY CXX_STANDARD 20)

cmake_minimum_required(VERSION 3.10)
project(slam)
set(OPENCV_EXTRA_MODULES_PATH /home/alan/opencv_build/opencv_contrib/modules)

set(CMAKE_PREFIX_PATH /home/alan/Downloads/libtorch)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INSTALL_PREFIX /usr/local)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)


# The first thing do is to tell cmake to find the TorchVision library.
# The package pulls in all the necessary torch libraries,
# so there is no need to also add `find_package(Torch)` here.

find_package( OpenCV REQUIRED )
find_package(TorchVision REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(slam src/main.cpp
           include/dataset.h
           src/dataset.cpp)

target_include_directories(slam PRIVATE include)


# We now need to link the TorchVision library to our executable.
# We can do that by using the TorchVision::TorchVision target,
# which also adds all the necessary torch dependencies.

target_compile_features(slam PUBLIC cxx_range_for)
target_link_libraries( slam ${OpenCV_LIBS})
target_link_libraries(slam TorchVision::TorchVision)
set_property(TARGET slam PROPERTY CXX_STANDARD 17)



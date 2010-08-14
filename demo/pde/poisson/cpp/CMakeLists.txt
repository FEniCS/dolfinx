# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

# Set verbose output while testing CMake
set(CMAKE_VERBOSE_MAKEFILE 1)

# Get DOLFIN configuration data (dolfin-config.cmake must be in CMAKE_MODULE_PATH)
find_package(dolfin PATHS $ENV{CMAKE_MODULE_PATH})

# DOLFIN include directories
include_directories (${DOLFIN_INCLUDE_DIRS})

# DOLFIN library directories
link_directories(${DOLFIN_LIBRARY_DIR})

# Executbale
add_executable(demo main.cpp)

# DOLFIN library
target_link_libraries(demo ${DOLFIN_LIBRARIES})


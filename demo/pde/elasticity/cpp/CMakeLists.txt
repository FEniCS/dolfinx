# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

project(elasticity-demo)

# Set verbose output while testing CMake
set(CMAKE_VERBOSE_MAKEFILE 1)

# Get DOLFIN configuration data (dolfin-config.cmake must be in DOLFIN_CMAKE_CONFIG_PATH)
find_package(dolfin)

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug MinSizeRel Release RelWithDebInfo." FORCE)
endif()

# Compiler definitions
add_definitions(${DOLFIN_CXX_DEFINITIONS})

# Add special DOLFIN compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DOLFIN_CXX_FLAGS}")

# DOLFIN include directories
include_directories (${DOLFIN_INCLUDE_DIRS})

# DOLFIN library directories
link_directories(${DOLFIN_LIBRARY_DIR})

# Executable
add_executable(elasticity-demo main.cpp)

# DOLFIN library
target_link_libraries(elasticity-demo ${DOLFIN_LIBRARIES})

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

# Permit user to set location of the config file
#set(CMAKE_MODULE_PATH CACHE STRING "User-specified location of the DOLFIN CMake configure file" )

# Location of DOLFIN CMake config file (could also set CMAKE_MODULE_PATH)
#set(dolfin_DIR $ENV{DOLFIN_CMAKE_CONFIG_PATH})
#set(CMAKE_MODULE_PATHH $ENV{DOLFIN_CMAKE_CONFIG_PATH})
#message(STATUS "Test ${dolfin_DIR}")

# Set verbose output while testing CMake
set(CMAKE_VERBOSE_MAKEFILE 1)

# Get DOLFIN configuration data (dolfin-config.cmake must be in DOLFIN_CMAKE_CONFIG_PATH)
find_package(dolfin)

# Compiler definitions
add_definitions(${DOLFIN_CXX_DEFINITIONS})

# DOLFIN include directories
include_directories (${DOLFIN_INCLUDE_DIRS})

# DOLFIN library directories
link_directories(${DOLFIN_LIBRARY_DIR})

# Executable
add_executable(method-weights-demo main.cpp)

# DOLFIN library
target_link_libraries(method-weights-demo ${DOLFIN_LIBRARIES})

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

project(conditional-demo)

# Permit user to set location of the config file
#set(CMAKE_MODULE_PATH CACHE STRING "User-specified location of the DOLFIN CMake configure file" )

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

# DOLFIN include directories
include_directories (${DOLFIN_INCLUDE_DIRS})

# DOLFIN library directories
link_directories(${DOLFIN_LIBRARY_DIR})

# Executable
add_executable(conditional-demo main.cpp)

# DOLFIN library
target_link_libraries(conditional-demo ${DOLFIN_LIBRARIES})

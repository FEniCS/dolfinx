# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

# Get DOLFIN configuration data (dolfin-config.cmake must be in DOLFIN_CMAKE_CONFIG_PATH)
find_package(dolfin)

# Compiler definitions
add_definitions(${DOLFIN_CXX_DEFINITIONS})

# DOLFIN include directories
include_directories (${DOLFIN_INCLUDE_DIRS})

# DOLFIN library directories
link_directories(${DOLFIN_LIBRARY_DIR})

# Executable
add_executable(poisson-demo main.cpp)

# DOLFIN library
target_link_libraries(poisson-demo ${DOLFIN_LIBRARIES})

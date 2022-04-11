cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME interpolation)
project(${PROJECT_NAME})

# Get DOLFINX configuration data (DOLFINXConfig.cmake must be in
# DOLFINX_CMAKE_CONFIG_PATH)
if (NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
  find_package(Basix REQUIRED)
endif()

# Executable
add_executable(${PROJECT_NAME}
                main.cpp
                )

# Set C++17 standard
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx)

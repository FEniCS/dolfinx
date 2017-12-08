cmake_minimum_required(VERSION 3.5)
project(dolfin-tests)

find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

# Prepare "Catch" library for other executables
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/catch)

add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

# Make test executable
set(TEST_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/common/SubSystemsManager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/function/Expression.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/geometry/ConvexTriangulation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/io/XMLMeshData.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/io/XMLMeshValueCollection.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/la/LinearOperator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/la/Vector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh/Mesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh/MeshColoring.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh/MeshFunction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh/MeshValueCollection.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mesh/MultiMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/parameter/Parameters.cpp
  )

add_executable(unittests ${TEST_SOURCES})
target_link_libraries(unittests PRIVATE Catch dolfin)

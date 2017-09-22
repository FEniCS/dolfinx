cmake_minimum_required(VERSION 3.5)
project(dolfin-tests)

find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

# Prepare "Catch" library for other executables
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

# Make test executable
set(TEST_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/common/SubSystemsManager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/function/Expression.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/io/XMLMeshData.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/io/XMLMeshValueCollection.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/la/LinearOperator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/la/Vector.cpp
  )

add_executable(unittests_cpp ${TEST_SOURCES})
target_link_libraries(unittests_cpp PRIVATE Catch dolfin)

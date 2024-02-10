# ------------------------------------------------------------------------------
# Top level CMakeLists.txt file for DOLFINx
cmake_minimum_required(VERSION 3.19)

# ------------------------------------------------------------------------------
# Set project name and version number
project(DOLFINX_DOCUMENTATION VERSION "0.8.0.0")

# ------------------------------------------------------------------------------
# Use C++20
set(CMAKE_CXX_STANDARD 20)

# Require C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Do not enable compler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# Get GIT changeset, if available
find_program(GIT_FOUND git)

if(GIT_FOUND)
  # Get the commit hash of the working branch
  execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
else()
  set(GIT_COMMIT_HASH "unknown")
endif()

message(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

cmake_path(GET PROJECT_SOURCE_DIR PARENT_PATH DOLFINX_CPP_DIR)

message(DOLFINX_CPP_DIR="${DOLFINX_CPP_DIR}")

set(demos ${DOLINX_CPP_DIR}/demo)

# Add the cmake folder so the FindSphinx module is found
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/source" ${CMAKE_MODULE_PATH})

find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

set(DOXYGEN_INPUT_DIR ${DOLFINX_CPP_DIR}/dolfinx)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# Set the Doxygen input and output directories in the Doxyfile
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

# Create output directory for Doxygen
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) 

add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                   COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                   MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                   COMMENT "Generating docs"
                   VERBATIM)

# Generate Doxygen doc on dolfinx and demo
add_custom_target(Doxygen DEPENDS ${DOXYGEN_INDEX_FILE})

# Generate Sphinx from Doxygen doc of dolfinx and demo
set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/source)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
                   COMMAND ${SPHINX_EXECUTABLE} -b html
                   # Tell Breathe where to find the Doxygen output
                   -Dbreathe_projects.DOLFINx=${DOXYGEN_OUTPUT_DIR}/xml
                   ${SPHINX_SOURCE} ${SPHINX_BUILD}
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   DEPENDS
                   ${CMAKE_CURRENT_SOURCE_DIR}/source/index.rst
                   ${DOXYGEN_INDEX_FILE}
                   MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
                   COMMENT "Generating documentation with Sphinx")

add_custom_target(Sphinx DEPENDS ${SPHINX_INDEX_FILE})

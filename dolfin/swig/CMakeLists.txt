project(swig)

# Remove '-Werror', '-Wall' and 'pedantic' flags (if present) when
# compiling SWIG-generated files
string(REGEX REPLACE "-Wall" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-Wall" " " CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER}")
string(REGEX REPLACE "-Werror=format-security" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-Werror=format-security" " " CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER}")
string(REGEX REPLACE "-Werror" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-Werror" " " CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER}")
string(REGEX REPLACE "-pedantic" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-pedantic" " " CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER}")

# Prevent Intel compiler emitting warnings for SWIG-generated code
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  if("${CMAKE_CXX_COMPILER_VERSION}" LESS "13")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd186")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -wd186")
    set(CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER} -wd186")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -wd186")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -wd186")
  endif()
endif()

# Prevent clang emitting warnings for SWIG-generated code
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  set(CLANG_FLAGS "-Wno-logical-not-parentheses -Wno-logical-op-parentheses -Wno-tautological-constant-out-of-range-compare")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CLANG_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CLANG_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER} ${CLANG_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CLANG_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CLANG_FLAGS}")
endif()

# Set common include directoris
include_directories(${NUMPY_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS} ${CMAKE_BINARY_DIR})

# Set add PYTHON defines
add_definitions(${DOLFIN_PYTHON_DEFINITIONS})

# Check that swig interface is generated, otherwise re-generate it
if (NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/modules)
  message(STATUS "")
  message(STATUS "Generating SWIG interface.")
  message(STATUS "--------------------------")
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${DOLFIN_SOURCE_DIR}/cmake/scripts/generate-swig-interface.py ${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${DOLFIN_SOURCE_DIR})
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${DOLFIN_SOURCE_DIR}/cmake/scripts/generate-swig-docstrings.py ${CMAKE_BINARY_DIR}
    WORKING_DIRECTORY ${DOLFIN_SOURCE_DIR})
endif()

# Get combined modules
file(GLOB COMBINED_MODULES RELATIVE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/modules/*)

# Sort and reverse the list so la combined module comes early
list(SORT COMBINED_MODULES)
list(REVERSE COMBINED_MODULES)

# Get all original interface files
file(GLOB_RECURSE ORIGINAL_INTERFACE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.i)

# Target to copy all swig files from source to build directory
add_custom_target(copy_swig_files
  COMMAND ${PYTHON_EXECUTABLE} ${DOLFIN_SOURCE_DIR}/cmake/scripts/copy-swig-files.py ${CMAKE_BINARY_DIR}
  DEPENDS ${ORIGINAL_INTERFACE_FILES}
  )

# Target to generate all swig interface files
add_custom_target(generate_swig_files
  COMMAND ${PYTHON_EXECUTABLE} ${DOLFIN_SOURCE_DIR}/cmake/scripts/generate-swig-interface.py ${CMAKE_BINARY_DIR}
  COMMAND ${PYTHON_EXECUTABLE} ${DOLFIN_SOURCE_DIR}/cmake/scripts/generate-swig-docstrings.py ${CMAKE_BINARY_DIR}
  WORKING_DIRECTORY ${DOLFIN_SOURCE_DIR}
  COMMENT "Generating all SWIG interface files"
  )

# Target to generate swig interface files
add_custom_target(generate_swig_interface
  COMMAND ${PYTHON_EXECUTABLE} ${DOLFIN_SOURCE_DIR}/cmake/scripts/generate-swig-interface.py ${CMAKE_BINARY_DIR}
  WORKING_DIRECTORY ${DOLFIN_SOURCE_DIR}
  COMMENT "Generating SWIG interface files"
  )

# Target to generate swig docstring files
add_custom_target(generate_swig_docstrings
  COMMAND ${PYTHON_EXECUTABLE} ${DOLFIN_SOURCE_DIR}/cmake/scripts/generate-swig-docstrings.py ${CMAKE_BINARY_DIR}
  WORKING_DIRECTORY ${DOLFIN_SOURCE_DIR}
  COMMENT "Generating SWIG docstring files from C++ headers"
  )

# Target to generate swig docstring files with debug
add_custom_target(generate_swig_docstrings_with_debug
  COMMAND ${PYTHON_EXECUTABLE} ${DOLFIN_SOURCE_DIR}/cmake/scripts/generate-swig-docstrings.py ${CMAKE_BINARY_DIR} DEBUG
  WORKING_DIRECTORY ${DOLFIN_SOURCE_DIR}
  COMMENT "Generating SWIG docstring files from C++ headers (DEBUG modus)"
  )

# Iterate over the combined modules and add each as a target
foreach(COMBINED_MODULE_DIR ${COMBINED_MODULES})
  add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/${COMBINED_MODULE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/${COMBINED_MODULE_DIR})
endforeach()

# List of files which should not be installed
set(NOT_INSTALL )

# Iterate over each sub module and add install targets
foreach(MODULE_DIR ${DOLFIN_DIRS})

  # Install pre.i if exists
  if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_DIR}/pre.i)
    install(FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_DIR}/pre.i
      DESTINATION
      ${DOLFIN_INCLUDE_DIR}/dolfin/swig/${MODULE_DIR}
      COMPONENT Development)
  endif()

endforeach()

# Add global interface files
file(GLOB GLOBAL_INTERFACE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.i)
install(FILES ${GLOBAL_INTERFACE_FILES}
  DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/swig
  COMPONENT Development
  )

# Add global typemap interface files
file(GLOB TYPEMAP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/typemaps/*.i)
install(FILES ${TYPEMAP_FILES}
  DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/swig/typemaps
  COMPONENT Development
  )

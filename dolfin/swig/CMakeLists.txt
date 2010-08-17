project(cpp)

#SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Make sure we can import the UFC module and that it is
# compiled with present version of SWIG
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import ufc, sys; sys.stdout.write('.'.join([str(int(s)) for s in ufc.__swigversion__.split('.')]))"
  OUTPUT_VARIABLE UFC_SWIGVERSION
  RESULT_VARIABLE UFC_MODULE_NOT_FOUND
  )
if(UFC_MODULE_NOT_FOUND)
  message(FATAL_ERROR "Unable to import UFC. Install latest UFC.")
endif(UFC_MODULE_NOT_FOUND)
if(NOT "${SWIG_VERSION}" STREQUAL "${UFC_SWIGVERSION}")
  message(FATAL_ERROR "UFC compiled with different version of SWIG. Please install SWIG version ${UFC_SWIGVERSION} or recompile UFC with present SWIG.")
endif(NOT "${SWIG_VERSION}" STREQUAL "${UFC_SWIGVERSION}")


get_directory_property(cmake_defs COMPILE_DEFINITIONS)
set(SWIG_MODULE_NAME cpp)
set(CMAKE_SWIG_FLAGS
  -module ${SWIG_MODULE_NAME}
  -shadow
  -modern
  -modernargs
  -fastdispatch
  -fvirtual
  -nosafecstrings
  -noproxydel
  -fastproxy
  -fastinit
  -fastunpack
  -fastquery
  -nobuildnone
  -Iinclude/swig
  ${DOLFIN_CXX_DEFINITIONS}
  )
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR})

set(SWIG_SOURCES dolfin.i)
set_source_files_properties(${SWIG_SOURCES} PROPERTIES CPLUSPLUS ON)

# Remove '-Werror' and '-Wall' flags (if present) when compiling SWIG-generated files
string(REGEX REPLACE "-Wall" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-Wall" " " CMAKE_CXX_FLAGS_MAINTAINER "${CMAKE_CXX_FLAGS_MAINTAINER}")
string(REGEX REPLACE "-Werror" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-Werror" " " CMAKE_CXX_FLAGS_MAINTAINER "${CMAKE_CXX_FLAGS_MAINTAINER}")

# The prevents swig being run unnecessarily
set_source_files_properties(dolfin.i PROPERTIES SWIG_MODULE_NAME cpp)

include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})

swig_add_module(${SWIG_MODULE_NAME} python ${SWIG_SOURCES})
swig_link_libraries(cpp dolfin ${PYTHON_LIBRARIES})

get_target_property(SWIG_MODULE_LOCATION ${SWIG_MODULE_cpp_REAL_NAME} LOCATION)
install(FILES
  ${SWIG_MODULE_LOCATION} ${CMAKE_CURRENT_BINARY_DIR}/${SWIG_MODULE_NAME}.py
  DESTINATION ${DOLFIN_INSTALL_PYTHON_MODULE_DIR}/dolfin
  COMPONENT RuntimeLibraries
  )

install(DIRECTORY
  ${CMAKE_SOURCE_DIR}/site-packages/dolfin
  DESTINATION ${DOLFIN_INSTALL_PYTHON_MODULE_DIR}
  USE_SOURCE_PERMISSIONS
  COMPONENT RuntimeLibraries
  )

file(GLOB DOLFIN_SWIG_INTERFACE_FILES "*.i")
install(FILES ${DOLFIN_SWIG_INTERFACE_FILES}
  DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/swig
  COMPONENT Development
  )

install(DIRECTORY "import"
  DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/swig
  USE_SOURCE_PERMISSIONS
  COMPONENT Development
  )

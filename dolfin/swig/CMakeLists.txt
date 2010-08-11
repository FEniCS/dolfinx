project(cpp)

SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Make sure UFC is compiled with present version of SWIG
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import sys, ufc; sys.stdout.write(ufc.__swigversion__)"
  OUTPUT_VARIABLE UFC_SWIGVERSION
  )
if(NOT "${SWIG_VERSION}" STREQUAL "${UFC_SWIGVERSION}")
  message(FATAL_ERROR "UFC compiled with different version of SWIG. Please install SWIG version ${UFC_SWIGVERSION} or recompile UFC with present SWIG.")
endif(NOT "${SWIG_VERSION}" STREQUAL "${UFC_SWIGVERSION}")

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
  )
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR})

set(SWIG_SOURCES dolfin.i)
set_source_files_properties(${SWIG_SOURCES} PROPERTIES CPLUSPLUS ON)

include_directories(${PYTHON_INCLUDE_PATH})

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

project(cpp)

#SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

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

# Get all SWIG interface files
file(GLOB DOLFIN_SWIG_INTERFACE_FILES "*.i")

# and the DOLFIN header files
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import sys, re; sys.stdout.write(';'.join([r'${DOLFIN_SOURCE_DIR}/'+re.findall('dolfin.*\\.h', l)[0] for l in open(r'${DOLFIN_SOURCE_DIR}/'+'dolfin/swig/kernel_modules.i').readlines() if '%include' in l and not 'swig' in l]))"
  OUTPUT_VARIABLE SWIG_KERNEL_MODULES
)

# Set SWIG source file
set(SWIG_SOURCES dolfin.i)
set_source_files_properties(${SWIG_SOURCES} PROPERTIES CPLUSPLUS ON)

# Remove '-Werror', '-Wall' and 'pedantic' flags (if present) when compiling SWIG-generated files
string(REGEX REPLACE "-Wall" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-Wall" " " CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER}")
string(REGEX REPLACE "-Werror=format-security" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-Werror=format-security" " " CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER}")
string(REGEX REPLACE "-Werror" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-Werror" " " CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER}")
string(REGEX REPLACE "-pedantic" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-pedantic" " " CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER}")

# The prevents swig being run unnecessarily
set_source_files_properties(dolfin.i PROPERTIES SWIG_MODULE_NAME cpp)

# Tell CMake which SWIG interface files should be checked for changes when recompile
set(SWIG_MODULE_cpp_EXTRA_DEPS ${DOLFIN_SWIG_INTERFACE_FILES})
list(APPEND SWIG_MODULE_cpp_EXTRA_DEPS ${SWIG_KERNEL_MODULES})

include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
swig_add_module(${SWIG_MODULE_NAME} python ${SWIG_SOURCES})
swig_link_libraries(cpp dolfin ${PYTHON_LIBRARIES})

get_target_property(SWIG_MODULE_LOCATION ${SWIG_MODULE_cpp_REAL_NAME} LOCATION)

# Install Python .py files
install(FILES
  ${SWIG_MODULE_LOCATION} ${CMAKE_CURRENT_BINARY_DIR}/${SWIG_MODULE_NAME}.py
  DESTINATION ${DOLFIN_INSTALL_PYTHON_MODULE_DIR}/dolfin
  COMPONENT RuntimeLibraries
  )

install(FILES ${DOLFIN_SWIG_INTERFACE_FILES}
  DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/swig
  COMPONENT Development
  )

install(DIRECTORY "import"
  DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/swig
  USE_SOURCE_PERMISSIONS
  COMPONENT Development
  )

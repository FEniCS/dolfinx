# Automatic get the module name
get_filename_component(SWIG_MODULE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Set project name
project(${SWIG_MODULE_NAME})

# What does this do?
get_directory_property(cmake_defs COMPILE_DEFINITIONS)

# Set SWIG flags 
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

# Get all SWIG interface files
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/interface_files.txt DOLFIN_SWIG_INTERFACE_FILES )

# and the DOLFIN header files
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/headers.txt SWIG_KERNEL_MODULES)

# This prevents swig being run unnecessarily
set_source_files_properties(module.i PROPERTIES SWIG_MODULE_NAME ${SWIG_MODULE_NAME})

# Tell CMake SWIG has generated a C++ file
set_source_files_properties(module.i PROPERTIES CPLUSPLUS ON)

# Generate SWIG files in 
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR})

# Tell CMake which SWIG interface files should be checked for changes when recompile
set(SWIG_MODULE_${SWIG_MODULE_NAME}_EXTRA_DEPS ${DOLFIN_SWIG_INTERFACE_FILES} ${SWIG_KERNEL_MODULES})

# Tell CMake to run SWIG on module.i and to link against libdolfin
swig_add_module(${SWIG_MODULE_NAME} python module.i)
swig_link_libraries(${SWIG_MODULE_NAME} dolfin ${PYTHON_LIBRARIES})

# Install Python .py files
get_target_property(SWIG_MODULE_LOCATION ${SWIG_MODULE_${SWIG_MODULE_NAME}_REAL_NAME} LOCATION)
install(FILES
  ${SWIG_MODULE_LOCATION} ${CMAKE_CURRENT_BINARY_DIR}/${SWIG_MODULE_NAME}.py
  DESTINATION ${DOLFIN_INSTALL_PYTHON_MODULE_DIR}/dolfin/cpp
  COMPONENT RuntimeLibraries
  )

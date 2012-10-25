project(swig)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Remove '-Werror', '-Wall' and 'pedantic' flags (if present) when compiling SWIG-generated files
string(REGEX REPLACE "-Wall" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-Wall" " " CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER}")
string(REGEX REPLACE "-Werror=format-security" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-Werror=format-security" " " CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER}")
string(REGEX REPLACE "-Werror" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-Werror" " " CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER}")
string(REGEX REPLACE "-pedantic" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "-pedantic" " " CMAKE_CXX_FLAGS_DEVELOPER "${CMAKE_CXX_FLAGS_DEVELOPER}")

# Set common include directoris
include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})

# Get combined modules
file(GLOB COMBINED_MODULES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/modules/*)

# Sort and reverse the list so la combined module comes early
list(SORT COMBINED_MODULES)
list(REVERSE COMBINED_MODULES)

# Iterate over the combined modules and add each as a target
foreach(COMBINED_MODULE_DIR ${COMBINED_MODULES})
  add_subdirectory(${COMBINED_MODULE_DIR})
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


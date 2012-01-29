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

# Iterate over the combined modules and add each as a target
foreach(COMBINED_MODULE_DIR ${COMBINED_MODULES})
  add_subdirectory(${COMBINED_MODULE_DIR})
endforeach()

# List of files which should not be installed
set(NOT_INSTALL )

# Iterate over each module and add install targets
foreach(MODULE_DIR ${DOLFIN_DIRS})
  file(GLOB INTERFACE_FILES RELATIVE ${MODULE_DIR} ${MODULE_DIR}/*.i)
  foreach(INTERFACE_FILE ${INTERFACE_FILES})
    get_filename_component(LOCAL_INTERFACE_FILE ${INTERFACE_FILE} NAME)
    if (NOT ${_DIR_NAME} STREQUAL ${DOLFIN_H} AND 
	NOT ${_DIR_NAME} STREQUAL swig AND 
	NOT ${_DIR_NAME} STREQUAL CMakeLists.txt )
      list(APPEND DOLFIN_DIRS ${_DIR_NAME})
    endif()
    message("INTERFACE_FILE: ${INTERFACE_FILE}")
  endforeach()
  install(FILES ${_HEADERS} DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/${DIR} COMPONENT Development)
endforeach()

#install(FILES ${ALL_INTERFACE_FILES}
#  DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/swig
#  COMPONENT Development
#  )

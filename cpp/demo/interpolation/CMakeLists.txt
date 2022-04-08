cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME interpolation)
project(${PROJECT_NAME})

# Get DOLFINX configuration data (DOLFINXConfig.cmake must be in
# DOLFINX_CMAKE_CONFIG_PATH)
if (NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
  find_package(Basix REQUIRED)
endif()

set(BLA_VENDOR OpenBLAS)
find_package(LAPACK)

set(FFCFLAGS CACHE INTERNAL "")

function(preprocess_ufl out_var)
  set(result)
  foreach(in_f ${ARGN})
      #    file(RELATIVE_PATH out_f ${CMAKE_CURRENT_SOURCE_DIR} ${in_f})
    string(REGEX REPLACE "\\.[^.]*$" "" in_f_noext ${in_f})
    set(out_f "${in_f_noext}.c")
    get_filename_component(in_f_name ${in_f} NAME)
    add_custom_command(OUTPUT ${out_f} ${in_f_noext}.h
      COMMAND ffcx ${FFCFLAGS} ${in_f_name} > /dev/null
      DEPENDS ${in_f}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Compiling py file to ${out_f}"
      VERBATIM
      )
    list(APPEND result ${out_f})
  endforeach()
  set(${out_var} "${result}" PARENT_SCOPE)
endfunction()

preprocess_ufl(ufl2cpp_files
    ${CMAKE_CURRENT_SOURCE_DIR}/interpolation.py
    )

# Executable
add_executable(${PROJECT_NAME}
                main.cpp
                ${ufl2cpp_files}
                )

# Set C++17 standard
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx)
target_link_libraries(${PROJECT_NAME} ${LAPACK_LIBRARIES})
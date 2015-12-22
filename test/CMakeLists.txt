project(dolfin-tests)

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

# FIXME: Temporary fix for whitespace error
cmake_policy(SET CMP0004 OLD)

# Set special link option, see `cmake --help-policy CMP0003`
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif()

# Find DOLFIN config file (not used here, but checks that tests will be able
# to find it
find_package(DOLFIN REQUIRED)

# If config file is found, add all demo sub-directories, else print helper
# message
if (DOLFIN_FOUND)

  # Build list of all cpp directories
  file(GLOB_RECURSE initial_list "*.cpp")

  set(added_dirs "")

  # Add each C++ demo directory only once
  foreach (cpp_test ${initial_list})

    # Get directory 
    get_filename_component(cpp_lib_dir ${cpp_test} DIRECTORY)

    # Get one upper directory
    get_filename_component(cpp_dir ${cpp_lib_dir} DIRECTORY)

    # Get last dir name
    get_filename_component(last_component ${cpp_dir} NAME)

    # Only process if last dirname is cpp
    if ((${last_component} STREQUAL "cpp"))

      # Check if the director is already added
      set(already_in "")
      foreach (dir ${added_dirs})
        if (${dir} STREQUAL ${cpp_lib_dir})
          set(already_in TRUE)
        endif()
      endforeach()

      # Schedule directory for compilation
      if (NOT already_in)
          list(APPEND added_dirs ${cpp_lib_dir})
          add_subdirectory(${cpp_lib_dir})
      endif()
    endif()
  endforeach()
else()

  message(STATUS "Could not locate DOLFINConfig.cmake file. Did you do 'make install' for the DOLFIN library and set the appropriate paths (source <build_dir>/dolfin.conf)?")

endif()

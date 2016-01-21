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

# Add GoogleTest Directory
include_directories(${CMAKE_BINARY_DIR}/GoogleTest/src/googletest/include)
link_directories(${CMAKE_BINARY_DIR}/GoogleTest/src/googletest-build)

# If config file is found, add all demo sub-directories, else print helper
# message
if (DOLFIN_FOUND)

  # Build list of all cpp directories
  file(GLOB_RECURSE initial_list "*.cpp")

  set(added_files "") # Change to targetting cpp files instead

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
      # If you find the cpp lib dir
      # That means you have allocated the cpp files
      # Don't need to recheck the directories
      list(APPEND added_files ${cpp_test})

    endif()
  endforeach()

  # Set CMake behavior
  cmake_policy(SET CMP0004 NEW)

  if (EXISTS ${DOLFIN_USE_FILE})
    include(${DOLFIN_USE_FILE})

    # Default build type (can be overridden by user)
    if (NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
        "Choose the type of build, options are: Debug MinSizeRel Release RelWithDebInfo." FORCE)
    endif()
  else()
    # Compiler definitions
    add_definitions(${DOLFIN_CXX_DEFINITIONS})

    # Compiler flags
    set(CMAKE_CXX_FLAGS "${DOLFIN_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

    # Include directories
    include_directories(${DOLFIN_INCLUDE_DIRS})
    include_directories(SYSTEM ${DOLFIN_3RD_PARTY_INCLUDE_DIRS})
  endif()

  add_executable(cpp_tests ${added_files})
  target_link_libraries(all_tests ${DOLFIN_LIBRARIES} gtest gtest_main)

else()

  message(STATUS "Could not locate DOLFINConfig.cmake file. Did you do 'make install' for the DOLFIN library and set the appropriate paths (source <build_dir>/dolfin.conf)?")

endif()

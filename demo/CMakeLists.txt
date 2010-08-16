#cmake_minimum_required(VERSION 2.8)

# Set verbose output while testing CMake
set(CMAKE_VERBOSE_MAKEFILE 1)

# Build list of all cpp demo directories
file( GLOB_RECURSE list "main.cpp")
list( SORT list )
string(REGEX REPLACE "/main.cpp" "" list "${list}")

#foreach( entry ${list} )
#  if ( IS_DIRECTORY ${entry} )
#    message(STATUS "CPP dir: ${entry}")
#  endif()
#endforeach()

# Add each C++ demo directory
foreach( demo_dir ${list} )
  message(STATUS "CPP dir: ${demo_dir}")
  add_subdirectory( ${demo_dir} )
endforeach()

# Manual list of C++ demo directories
#set(DOLFIN_DEMOS_DIRS
#  /home/garth/code/fenics/dolfin/dolfin-cmake/demo/pde/poisson/cpp
  #/home/garth/code/fenics/dolfin/dolfin-cmake/demo/pde/elasticity/cpp
  #/home/garth/code/fenics/dolfin/dolfin-cmake/demo/pde/simple/cpp
#  )

#foreach( demo_dir ${DOLFIN_DEMOS_DIRS} )
#  message(STATUS "Man CPP dir: ${demo_dir}")
#  add_subdirectory( ${demo_dir} )
#endforeach()




# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

project(demo_plot-qt)

set(SOURCES main.cpp Plotter.cpp PlotWidget.cpp CoordLabel.cpp BoundaryMeshFunction.cpp)
set(HEADERS Plotter.h PlotWidget.h CoordLabel.h BoundaryMeshFunction.h)

# Ignore whitespace in library paths
cmake_policy(SET CMP0004 OLD)

# Get DOLFIN configuration data (DOLFINConfig.cmake must be in DOLFIN_CMAKE_CONFIG_PATH)
find_package(DOLFIN REQUIRED)

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug MinSizeRel Release RelWithDebInfo." FORCE)
endif()

# Compiler definitions
add_definitions(${DOLFIN_CXX_DEFINITIONS})

# Add special DOLFIN compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DOLFIN_CXX_FLAGS}")

# Include directories
include_directories(${DOLFIN_INCLUDE_DIRS})
include_directories(SYSTEM ${DOLFIN_3RD_PARTY_INCLUDE_DIRS})

# Qt
find_package(Qt4)
if (QT_FOUND)
  include(${QT_USE_FILE})
  add_definitions(${QT_DEFINITIONS})
endif()

# VTK
find_package(VTK)
if (VTK_FOUND)
  include(${VTK_USE_FILE})
  find_library(QVTK_FOUND QVTK HINTS ${VTK_LIBRARY_DIRS})
endif()

# If built stand-alone (in demo directory), DOLFIN_ENABLE_FOO are not set. We
# have to parse the definitions to see if qvtk is enabled in DOLFIN.
get_directory_property(cmake_defs COMPILE_DEFINITIONS)
foreach(d ${cmake_defs})
  if (${d} STREQUAL HAS_QVTK)
    set(QVTK_ENABLED 1)
  endif()
endforeach()

if (QT_FOUND AND VTK_FOUND AND QVTK_ENABLED)

  # Define headers to run moc on. The OPTIONS ... works around a problem with moc+boost (QT bug #22829).
  qt4_wrap_cpp(HEADERS_MOC ${HEADERS} OPTIONS   -DBOOST_TT_HAS_OPERATOR_HPP_INCLUDED
                                                -DBOOST_TT_HAS_BIT_AND_ASSIGN_HPP_INCLUDED
                                                -DBOOST_TT_HAS_BIT_AND_HPP_INCLUDED
                                                -DBOOST_TT_HAS_BIT_OR_ASSIGN_HPP_INCLUDED
                                                -DBOOST_TT_HAS_BIT_OR_HPP_INCLUDED
                                                -DBOOST_TT_HAS_BIT_XOR_ASSIGN_HPP_INCLUDED
                                                -DBOOST_TT_HAS_BIT_XOR_HPP_INCLUDED
                                                -DBOOST_TT_HAS_DIVIDES_ASSIGN_HPP_INCLUDED
                                                -DBOOST_TT_HAS_DIVIDES_HPP_INCLUDED
                                                -DBOOST_TT_HAS_EQUAL_TO_HPP_INCLUDED
                                                -DBOOST_TT_HAS_GREATER_EQUAL_HPP_INCLUDED
                                                -DBOOST_TT_HAS_GREATER_HPP_INCLUDED
                                                -DBOOST_TT_HAS_LEFT_SHIFT_ASSIGN_HPP_INCLUDED
                                                -DBOOST_TT_HAS_LEFT_SHIFT_HPP_INCLUDED
                                                -DBOOST_TT_HAS_LESS_EQUAL_HPP_INCLUDED
                                                -DBOOST_TT_HAS_LESS_HPP_INCLUDED
                                                -DBOOST_TT_HAS_LOGICAL_AND_HPP_INCLUDED
                                                -DBOOST_TT_HAS_LOGICAL_OR_HPP_INCLUDED
                                                -DBOOST_TT_HAS_MINUS_ASSIGN_HPP_INCLUDED
                                                -DBOOST_TT_HAS_MINUS_HPP_INCLUDED
                                                -DBOOST_TT_HAS_MODULUS_ASSIGN_HPP_INCLUDED
                                                -DBOOST_TT_HAS_MODULUS_HPP_INCLUDED
                                                -DBOOST_TT_HAS_MULTIPLIES_ASSIGN_HPP_INCLUDED
                                                -DBOOST_TT_HAS_MULTIPLIES_HPP_INCLUDED
                                                -DBOOST_TT_HAS_NOT_EQUAL_TO_HPP_INCLUDED
                                                -DBOOST_TT_HAS_PLUS_ASSIGN_HPP_INCLUDED
                                                -DBOOST_TT_HAS_PLUS_HPP_INCLUDED
                                                -DBOOST_TT_HAS_RIGHT_SHIFT_ASSIGN_HPP_INCLUDED
                                                -DBOOST_TT_HAS_RIGHT_SHIFT_HPP_INCLUDED
                                                )

  # Executable
  add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS_MOC})

  # Target libraries
  target_link_libraries(${PROJECT_NAME} ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${QT_LIBRARIES})

else()

  message(STATUS "QT not found, or QT/VTK not enabled in DOLFIN. Not building " ${PROJECT_NAME})

endif()

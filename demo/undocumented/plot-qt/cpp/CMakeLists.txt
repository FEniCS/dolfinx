# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

project(demo_plot-qt)

set(SOURCES main.cpp Plotter.cpp PlotWidget.cpp CoordLabel.cpp BoundaryMeshFunction.cpp)
set(HEADERS Plotter.h PlotWidget.h CoordLabel.h BoundaryMeshFunction.h)

# Ignore whitespace in library paths
cmake_policy(SET CMP0004 OLD)

# Get DOLFIN configuration data (dolfin-config.cmake must be in DOLFIN_CMAKE_CONFIG_PATH)
find_package(dolfin REQUIRED)

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

if (DOLFIN_ENABLE_QT AND QT_FOUND AND DOLFIN_ENABLE_VTK AND QVTK_FOUND)

  # Define headers to run moc on
  qt4_wrap_cpp(HEADERS_MOC ${HEADERS})

  # Executable
  add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS_MOC})

  # Target libraries
  target_link_libraries(${PROJECT_NAME} ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES} ${QT_LIBRARIES})

else()

  message(WARNING "Qt / QVTK not enabled, not building " ${PROJECT_NAME})

endif()


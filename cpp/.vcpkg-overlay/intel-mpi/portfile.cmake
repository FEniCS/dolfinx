set(INTELMPI_VERSION "2021.12")
set(SOURCE_PATH "${CURRENT_BUILDTREES_DIR}/src/intel-mpi-${INTELMPI_VERSION}")

cmake_path(SET SDK_SOURCE_DIR "C:/Program Files (x86)/Intel/oneAPI")

message(STATUS "Using Intel MPI source SDK at ${SDK_SOURCE_DIR}")

set(SDK_SOURCE_MPI_DIR "${SDK_SOURCE_DIR}/mpi/${INTELMPI_VERSION}")

set(SOURCE_INCLUDE_PATH "${SDK_SOURCE_MPI_DIR}/include")
set(SOURCE_LIB_PATH "${SDK_SOURCE_MPI_DIR}/lib")
set(SOURCE_DEBUG_LIB_PATH "${SDK_SOURCE_MPI_DIR}/lib/mpi/debug")
set(SOURCE_BIN_PATH "${SDK_SOURCE_MPI_DIR}/bin")
set(SOURCE_DEBUG_BIN_PATH "${SDK_SOURCE_MPI_DIR}/bin/mpi/debug")
set(SOURCE_TOOLS_PATH "${SDK_SOURCE_MPI_DIR}/bin")
set(SOURCE_LIBFABRIC_PATH "${SDK_SOURCE_MPI_DIR}/opt/mpi/libfabric/bin")

# Get files in include directory
file(
  GLOB_RECURSE SOURCE_INCLUDE_FILES
  LIST_DIRECTORIES TRUE
  "${SOURCE_INCLUDE_PATH}/*"
)

# Get files in bin directory
file(GLOB TOOLS_FILES "${SOURCE_TOOLS_PATH}/*.exe" "${SOURCE_TOOLS_PATH}/*.dll"
     "${SOURCE_TOOLS_PATH}/*.bat"
)

# Install tools files
file(INSTALL ${TOOLS_FILES} DESTINATION "${CURRENT_PACKAGES_DIR}/tools/${PORT}")

# Also install include files in the tools directory because the compiler
# wrappers (mpicc.bat for example) needs them
file(INSTALL ${SOURCE_INCLUDE_FILES}
     DESTINATION "${CURRENT_PACKAGES_DIR}/tools/${PORT}/include"
)

# Install include files
file(INSTALL ${SOURCE_INCLUDE_FILES}
     DESTINATION "${CURRENT_PACKAGES_DIR}/include"
)

# Install release library files
file(INSTALL "${SOURCE_LIB_PATH}/impi.lib" "${SOURCE_LIB_PATH}/impicxx.lib"
     DESTINATION "${CURRENT_PACKAGES_DIR}/lib"
)

# Install debug library files
file(INSTALL "${SOURCE_DEBUG_LIB_PATH}/impi.lib"
     "${SOURCE_DEBUG_LIB_PATH}/impicxx.lib"
     DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib"
)

# 'libfabric.dll' is not needed for the compilation but it is needed for the
# runtime and should be in the PATH for 'mpiexec' to work
file(INSTALL "${SOURCE_LIBFABRIC_PATH}/libfabric.dll"
     "${SOURCE_BIN_PATH}/impi.dll" "${SOURCE_BIN_PATH}/impi.pdb"
     DESTINATION "${CURRENT_PACKAGES_DIR}/bin"
)

file(INSTALL "${SOURCE_LIBFABRIC_PATH}/libfabric.dll"
     "${SOURCE_DEBUG_BIN_PATH}/impi.dll" "${SOURCE_DEBUG_BIN_PATH}/impi.pdb"
     DESTINATION "${CURRENT_PACKAGES_DIR}/debug/bin"
)

file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/mpi-wrapper.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
)

# Handle copyright
file(
  COPY "${SDK_SOURCE_DIR}/licensing/2024.1/licensing/2024.1/license.htm"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
)
file(WRITE "${CURRENT_PACKAGES_DIR}/share/${PORT}/copyright"
     "See the licence.htm file in this directory."
)

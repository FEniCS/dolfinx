set(HEADERS
  ALE.h
  dolfin_ale.h
  HarmonicSmoothing.h
  MeshDisplacement.h
  PARENT_SCOPE)

set(SOURCES
  ALE.cpp
  HarmonicSmoothing.cpp
  MeshDisplacement.cpp
  PARENT_SCOPE)

# Add custom command for compiling Poisson1D.ufl
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/Poisson1D.h
  MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/Poisson1D.ufl
  COMMAND ${PYTHON_EXECUTABLE} -m ffc -l dolfin -O -r auto -o ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/Poisson1D.ufl
  COMMENT "Compiling form Poisson1D.ufl")

# Add custom command for compiling Poisson2D.ufl
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/Poisson2D.h
  MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/Poisson2D.ufl
  COMMAND ${PYTHON_EXECUTABLE} -m ffc -l dolfin -O -r auto -o ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/Poisson2D.ufl
  COMMENT "Compiling form Poisson2D.ufl")

# Add custom command for compiling Poisson3D.ufl
add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/Poisson3D.h
  MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/Poisson3D.ufl
  COMMAND ${PYTHON_EXECUTABLE} -m ffc -l dolfin -O -r auto -o ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/Poisson3D.ufl
  COMMENT "Compiling form Poisson3D.ufl")

# Add custom target for compiling forms
add_custom_target(compile_ale_forms DEPENDS
  ${CMAKE_BINARY_DIR}/Poisson1D.h
  ${CMAKE_BINARY_DIR}/Poisson2D.h
  ${CMAKE_BINARY_DIR}/Poisson3D.h)

# Tell CMake that Poisson*D.h won't be available until build time
set_source_files_properties(${CMAKE_BINARY_DIR}/Poisson1D.h PROPERTIES GENERATED 1)
set_source_files_properties(${CMAKE_BINARY_DIR}/Poisson2D.h PROPERTIES GENERATED 1)
set_source_files_properties(${CMAKE_BINARY_DIR}/Poisson3D.h PROPERTIES GENERATED 1)

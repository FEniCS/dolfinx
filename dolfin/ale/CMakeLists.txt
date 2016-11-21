set(DIR ale)

# Headers
set(DOLFIN_${DIR}_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/ALE.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_ale.h
  ${CMAKE_CURRENT_SOURCE_DIR}/HarmonicSmoothing.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshDisplacement.h
  )

# Install headers
install(FILES ${DOLFIN_${DIR}_HEADERS}
  DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/${DIR}
  COMPONENT Development)

# Append headers
set(DOLFIN_HEADERS ${DOLFIN_HEADERS} ${DOLFIN_${DIR}_HEADERS} PARENT_SCOPE)

# Append sources
set(DOLFIN_SOURCES
  ${DOLFIN_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/ALE.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HarmonicSmoothing.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshDisplacement.cpp
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

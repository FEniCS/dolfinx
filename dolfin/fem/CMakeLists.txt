set(DIR fem)

# Headers
set(DOLFIN_${DIR}_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/assemble.h
  ${CMAKE_CURRENT_SOURCE_DIR}/assemble_local.h
  ${CMAKE_CURRENT_SOURCE_DIR}/AssemblerBase.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Assembler.h
  ${CMAKE_CURRENT_SOURCE_DIR}/BasisFunction.h
  ${CMAKE_CURRENT_SOURCE_DIR}/DirichletBC.h
  ${CMAKE_CURRENT_SOURCE_DIR}/DiscreteOperators.h
  ${CMAKE_CURRENT_SOURCE_DIR}/DofMapBuilder.h
  ${CMAKE_CURRENT_SOURCE_DIR}/DofMap.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_fem.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Equation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/fem_utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FiniteElement.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Form.h
  ${CMAKE_CURRENT_SOURCE_DIR}/GenericDofMap.h
  ${CMAKE_CURRENT_SOURCE_DIR}/LinearTimeDependentProblem.h
  ${CMAKE_CURRENT_SOURCE_DIR}/LinearVariationalProblem.h
  ${CMAKE_CURRENT_SOURCE_DIR}/LinearVariationalSolver.h
  ${CMAKE_CURRENT_SOURCE_DIR}/LocalAssembler.h
  ${CMAKE_CURRENT_SOURCE_DIR}/LocalSolver.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMeshAssembler.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMeshDirichletBC.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMeshDofMap.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMeshForm.h
  ${CMAKE_CURRENT_SOURCE_DIR}/NonlinearVariationalProblem.h
  ${CMAKE_CURRENT_SOURCE_DIR}/NonlinearVariationalSolver.h
  ${CMAKE_CURRENT_SOURCE_DIR}/OpenMpAssembler.h
  ${CMAKE_CURRENT_SOURCE_DIR}/PointSource.h
  ${CMAKE_CURRENT_SOURCE_DIR}/solve.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SparsityPatternBuilder.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SystemAssembler.h
  ${CMAKE_CURRENT_SOURCE_DIR}/UFC.h
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
  ${CMAKE_CURRENT_SOURCE_DIR}/assemble.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/assemble_local.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/AssemblerBase.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Assembler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DirichletBC.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DiscreteOperators.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DofMapBuilder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DofMap.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Equation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fem_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FiniteElement.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Form.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LinearTimeDependentProblem.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LinearVariationalProblem.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LinearVariationalSolver.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LocalAssembler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LocalSolver.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMeshAssembler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMeshDirichletBC.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMeshDofMap.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMeshForm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/NonlinearVariationalProblem.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/NonlinearVariationalSolver.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OpenMpAssembler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/PointSource.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/solve.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SparsityPatternBuilder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SystemAssembler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/UFC.cpp
  PARENT_SCOPE)

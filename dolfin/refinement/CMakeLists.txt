set(DIR refinement)

# Headers
set(DOLFIN_${DIR}_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/BisectionRefinement1D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_refinement.h
  ${CMAKE_CURRENT_SOURCE_DIR}/LocalMeshCoarsening.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ParallelRefinement.h
  ${CMAKE_CURRENT_SOURCE_DIR}/PlazaRefinementND.h
  ${CMAKE_CURRENT_SOURCE_DIR}/refine.h
  ${CMAKE_CURRENT_SOURCE_DIR}/RegularCutRefinement.h
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
  ${CMAKE_CURRENT_SOURCE_DIR}/BisectionRefinement1D.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LocalMeshCoarsening.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ParallelRefinement.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/PlazaRefinementND.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/refine.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/RegularCutRefinement.cpp
  PARENT_SCOPE)

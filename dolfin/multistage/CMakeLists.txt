set(DIR multistage)

# Headers
set(DOLFIN_${DIR}_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_multistage.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiStageScheme.h
  ${CMAKE_CURRENT_SOURCE_DIR}/PointIntegralSolver.h
  ${CMAKE_CURRENT_SOURCE_DIR}/RKSolver.h
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
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiStageScheme.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/PointIntegralSolver.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/RKSolver.cpp
  PARENT_SCOPE)

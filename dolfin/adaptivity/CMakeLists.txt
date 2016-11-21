set(DIR adaptivity)

# Headers
set(DOLFIN_${DIR}_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/adapt.h
  ${CMAKE_CURRENT_SOURCE_DIR}/AdaptiveLinearVariationalSolver.h
  ${CMAKE_CURRENT_SOURCE_DIR}/AdaptiveNonlinearVariationalSolver.h
  ${CMAKE_CURRENT_SOURCE_DIR}/adaptivesolve.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_adaptivity.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ErrorControl.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Extrapolation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/GenericAdaptiveVariationalSolver.h
  ${CMAKE_CURRENT_SOURCE_DIR}/GoalFunctional.h
  ${CMAKE_CURRENT_SOURCE_DIR}/marking.h
  ${CMAKE_CURRENT_SOURCE_DIR}/TimeSeries.h
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
  ${CMAKE_CURRENT_SOURCE_DIR}/adapt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/AdaptiveLinearVariationalSolver.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/AdaptiveNonlinearVariationalSolver.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/adaptivesolve.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ErrorControl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Extrapolation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/GenericAdaptiveVariationalSolver.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/GoalFunctional.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/marking.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TimeSeries.cpp
  PARENT_SCOPE)

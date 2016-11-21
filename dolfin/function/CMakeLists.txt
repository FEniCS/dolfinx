set(DIR function)

# Headers
set(DOLFIN_${DIR}_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/assign.h
  ${CMAKE_CURRENT_SOURCE_DIR}/CoefficientAssigner.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Constant.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_function.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Expression.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FunctionAssigner.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FunctionAXPY.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Function.h
  ${CMAKE_CURRENT_SOURCE_DIR}/FunctionSpace.h
  ${CMAKE_CURRENT_SOURCE_DIR}/GenericFunction.h
  ${CMAKE_CURRENT_SOURCE_DIR}/LagrangeInterpolator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMeshCoefficientAssigner.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMeshFunction.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMeshFunctionSpace.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMeshSubSpace.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SpecialFacetFunction.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SpecialFunctions.h
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
  ${CMAKE_CURRENT_SOURCE_DIR}/assign.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CoefficientAssigner.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Constant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Expression.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FunctionAssigner.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FunctionAXPY.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Function.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FunctionSpace.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/GenericFunction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LagrangeInterpolator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMeshCoefficientAssigner.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMeshFunction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMeshFunctionSpace.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MultiMeshSubSpace.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SpecialFacetFunction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SpecialFunctions.cpp
  PARENT_SCOPE)

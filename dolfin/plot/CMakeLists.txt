set(DIR plot)

# Headers
set(DOLFIN_${DIR}_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_plot.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ExpressionWrapper.h
  ${CMAKE_CURRENT_SOURCE_DIR}/GenericVTKPlottable.h
  ${CMAKE_CURRENT_SOURCE_DIR}/plot.h
  ${CMAKE_CURRENT_SOURCE_DIR}/VTKPlottableDirichletBC.h
  ${CMAKE_CURRENT_SOURCE_DIR}/VTKPlottableGenericFunction1D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/VTKPlottableGenericFunction.h
  ${CMAKE_CURRENT_SOURCE_DIR}/VTKPlottableMeshFunction.h
  ${CMAKE_CURRENT_SOURCE_DIR}/VTKPlottableMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/VTKPlotter.h
  ${CMAKE_CURRENT_SOURCE_DIR}/VTKWindowOutputStage.h
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
  ${CMAKE_CURRENT_SOURCE_DIR}/ExpressionWrapper.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/plot.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/VTKPlottableDirichletBC.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/VTKPlottableGenericFunction1D.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/VTKPlottableGenericFunction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/VTKPlottableMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/VTKPlottableMeshFunction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/VTKPlotter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/VTKWindowOutputStage.cpp
  PARENT_SCOPE)

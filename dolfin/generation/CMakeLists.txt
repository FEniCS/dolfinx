set(DIR generation)

# Headers
set(DOLFIN_${DIR}_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/BoxMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_generation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IntervalMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/RectangleMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SphericalShellMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitCubeMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitDiscMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitHexMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitIntervalMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitQuadMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitSquareMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTetrahedronMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTriangleMesh.h
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
  ${CMAKE_CURRENT_SOURCE_DIR}/BoxMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/IntervalMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/RectangleMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SphericalShellMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitDiscMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitHexMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitQuadMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTetrahedronMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTriangleMesh.cpp
  PARENT_SCOPE)

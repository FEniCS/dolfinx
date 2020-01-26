set(HEADERS_generation
  ${CMAKE_CURRENT_SOURCE_DIR}/BoxMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_generation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IntervalMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/RectangleMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitDiscMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTetrahedronMesh.h
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTriangleMesh.h
  PARENT_SCOPE)

target_sources(dolfinx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/BoxMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/IntervalMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/RectangleMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitDiscMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTetrahedronMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitTriangleMesh.cpp
)

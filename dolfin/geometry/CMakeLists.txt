set(DIR geometry)

# Headers
set(DOLFIN_${DIR}_HEADERS
  ${CMAKE_CURRENT_SOURCE_DIR}/BoundingBoxTree1D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/BoundingBoxTree2D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/BoundingBoxTree3D.h
  ${CMAKE_CURRENT_SOURCE_DIR}/BoundingBoxTree.h
  ${CMAKE_CURRENT_SOURCE_DIR}/CollisionDetection.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_geometry.h
  ${CMAKE_CURRENT_SOURCE_DIR}/GenericBoundingBoxTree.h
  ${CMAKE_CURRENT_SOURCE_DIR}/intersect.h
  ${CMAKE_CURRENT_SOURCE_DIR}/IntersectionTriangulation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshPointIntersection.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Point.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SimplexQuadrature.h
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
  ${CMAKE_CURRENT_SOURCE_DIR}/BoundingBoxTree.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CollisionDetection.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/GenericBoundingBoxTree.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/intersect.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/IntersectionTriangulation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MeshPointIntersection.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Point.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SimplexQuadrature.cpp
  PARENT_SCOPE)

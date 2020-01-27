set(HEADERS_geometry
  ${CMAKE_CURRENT_SOURCE_DIR}/BoundingBoxTree.h
  ${CMAKE_CURRENT_SOURCE_DIR}/CollisionPredicates.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_geometry.h
  ${CMAKE_CURRENT_SOURCE_DIR}/GeometryPredicates.h
  ${CMAKE_CURRENT_SOURCE_DIR}/predicates.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
  PARENT_SCOPE)

target_sources(dolfinx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/BoundingBoxTree.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CollisionPredicates.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/GeometryPredicates.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/predicates.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)

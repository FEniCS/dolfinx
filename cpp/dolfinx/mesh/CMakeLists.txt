set(HEADERS_mesh
    ${CMAKE_CURRENT_SOURCE_DIR}/dolfinx_mesh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Mesh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Geometry.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Topology.h
    ${CMAKE_CURRENT_SOURCE_DIR}/MeshTags.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cell_types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/generation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/graphbuild.h
    ${CMAKE_CURRENT_SOURCE_DIR}/permutationcomputation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/topologycomputation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/EntityMap.h
    PARENT_SCOPE
)

target_sources(
  dolfinx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Topology.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/cell_types.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/graphbuild.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/permutationcomputation.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/topologycomputation.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)

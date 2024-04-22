set(HEADERS_graph
    ${CMAKE_CURRENT_SOURCE_DIR}/AdjacencyList.h
    ${CMAKE_CURRENT_SOURCE_DIR}/dolfinx_graph.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ordering.h
    ${CMAKE_CURRENT_SOURCE_DIR}/partitioners.h
    ${CMAKE_CURRENT_SOURCE_DIR}/partition.h
    PARENT_SCOPE
)

target_sources(
  dolfinx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ordering.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/partitioners.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/partition.cpp
)

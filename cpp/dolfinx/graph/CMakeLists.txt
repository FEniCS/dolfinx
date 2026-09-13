set(
  HEADERS_graph
  ${CMAKE_CURRENT_SOURCE_DIR}/AdjacencyList.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfinx_graph.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ordering.h
  ${CMAKE_CURRENT_SOURCE_DIR}/partitioners.h
  ${CMAKE_CURRENT_SOURCE_DIR}/partition.h
  ${CMAKE_CURRENT_SOURCE_DIR}/sfc.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
  PARENT_SCOPE
)

target_sources(
  dolfinx
  PRIVATE ordering.cpp partitioners.cpp partition.cpp sfc.cpp utils.cpp
)

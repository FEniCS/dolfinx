set(HEADERS_graph
  ${CMAKE_CURRENT_SOURCE_DIR}/AdjacencyList.h
  ${CMAKE_CURRENT_SOURCE_DIR}/boostordering.h
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_graph.h
  ${CMAKE_CURRENT_SOURCE_DIR}/kahip.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parmetis.h
  ${CMAKE_CURRENT_SOURCE_DIR}/partition.h
  ${CMAKE_CURRENT_SOURCE_DIR}/scotch.h
  PARENT_SCOPE)

target_sources(dolfinx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/boostordering.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kahip.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/parmetis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/partition.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scotch.cpp
)

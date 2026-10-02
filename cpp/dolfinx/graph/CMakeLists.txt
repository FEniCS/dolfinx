target_sources(
  dolfinx
  PRIVATE ordering.cpp partition.cpp partitioners.cpp sfc.cpp utils.cpp
)

target_sources(
  dolfinx
  PUBLIC
    FILE_SET HEADERS
      FILES
        AdjacencyList.h
        dolfinx_graph.h
        ordering.h
        partition.h
        partitioners.h
        sfc.h
        utils.h
)

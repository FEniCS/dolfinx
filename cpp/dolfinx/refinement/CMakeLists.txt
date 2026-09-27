target_sources(dolfinx PRIVATE plaza.cpp uniform.cpp utils.cpp)

target_sources(
  dolfinx
  PUBLIC
    FILE_SET HEADERS
      FILES
        dolfinx_refinement.h
        interval.h
        mark.h
        option.h
        plaza.h
        refine.h
        uniform.h
        utils.h
)

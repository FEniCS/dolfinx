set(HEADERS_refinement
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_refinement.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/PlazaRefinementND.h
  ${CMAKE_CURRENT_SOURCE_DIR}/refine.h
  PARENT_SCOPE)

target_sources(dolfinx PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/PlazaRefinementND.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/refine.cpp
)

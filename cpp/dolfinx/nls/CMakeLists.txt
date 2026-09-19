set(
  HEADERS_nls
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfinx_nls.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SNESSolver.h
  PARENT_SCOPE
)

target_sources(dolfinx PRIVATE SNESSolver.cpp)

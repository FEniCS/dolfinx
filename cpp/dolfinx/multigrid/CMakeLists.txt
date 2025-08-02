set(HEADERS_multigrid
    ${CMAKE_CURRENT_SOURCE_DIR}/inclusion.h
    PARENT_SCOPE
)

target_sources(
  dolfinx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inclusion.cpp
)

set(HEADERS_refinement
    ${CMAKE_CURRENT_SOURCE_DIR}/dolfinx_refinement.h
    ${CMAKE_CURRENT_SOURCE_DIR}/interval.h ${CMAKE_CURRENT_SOURCE_DIR}/plaza.h
    ${CMAKE_CURRENT_SOURCE_DIR}/refine.h ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
    PARENT_SCOPE
)

target_sources(
  dolfinx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/plaza.cpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)

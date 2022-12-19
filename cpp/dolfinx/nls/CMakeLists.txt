set(HEADERS_nls
    ${CMAKE_CURRENT_SOURCE_DIR}/dolfin_nls.h
    ${CMAKE_CURRENT_SOURCE_DIR}/NewtonSolver.h
    PARENT_SCOPE
)

target_sources(dolfinx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/NewtonSolver.cpp)

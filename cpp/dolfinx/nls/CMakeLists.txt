set(
  HEADERS_nls
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfinx_nls.h
  ${CMAKE_CURRENT_SOURCE_DIR}/NewtonSolver.h
  ${CMAKE_CURRENT_SOURCE_DIR}/NonlinearProblem.h
  PARENT_SCOPE
)

target_sources(dolfinx PRIVATE NewtonSolver.cpp NonlinearProblem.cpp)

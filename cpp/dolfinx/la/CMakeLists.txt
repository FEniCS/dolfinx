set(
  HEADERS_la
  ${CMAKE_CURRENT_SOURCE_DIR}/dolfinx_la.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MatrixCSR.h
  ${CMAKE_CURRENT_SOURCE_DIR}/matrix_csr_impl.h
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mattrans.h
  ${CMAKE_CURRENT_SOURCE_DIR}/SparsityPattern.h
  ${CMAKE_CURRENT_SOURCE_DIR}/superlu_dist.h
  ${CMAKE_CURRENT_SOURCE_DIR}/Vector.h
  ${CMAKE_CURRENT_SOURCE_DIR}/petsc.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/slepc.h
  PARENT_SCOPE
)

target_sources(
  dolfinx
  PRIVATE SparsityPattern.cpp superlu_dist.cpp petsc.cpp slepc.cpp
)

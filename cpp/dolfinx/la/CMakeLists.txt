target_sources(
  dolfinx
  PRIVATE SparsityPattern.cpp petsc.cpp slepc.cpp superlu_dist.cpp
)

target_sources(
  dolfinx
  PUBLIC
    FILE_SET HEADERS
      FILES
        MatrixCSR.h
        SparsityPattern.h
        Vector.h
        dolfinx_la.h
        matmul.h
        matrix_csr_impl.h
        mattrans.h
        petsc.h
        slepc.h
        superlu_dist.h
        utils.h
)

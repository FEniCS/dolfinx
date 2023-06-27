
#pragma once

#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>

/// Solve A.u = b with SuperLU_dist
/// @param comm MPI_Comm
/// @param Amat CSR Matrix, distributed by row and finalized
/// @param bvec RHS vector
/// @param uvec Solution vector
/// @param verbose Output diagnostic information to stdout
template <typename T>
int superlu_solver(MPI_Comm comm, const dolfinx::la::MatrixCSR<T>& Amat,
                   const dolfinx::la::Vector<T>& bvec,
                   dolfinx::la::Vector<T>& uvec, bool verbose = true);

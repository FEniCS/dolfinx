
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

template <typename T>
class SuperLUSolver
{
public:
  SuperLUSolver(MPI_Comm comm, bool verbose = false);

  ~SuperLUSolver();

  void set_operator(const la::MatrixCSR<T>& Amat);

  /// Solve A.u=b
  /// @param b RHS Vector
  /// @param u Solution Vector
  /// @note Must be compatible with A
  int solve(const la::Vector<T>& b, la::Vector<T>& u);

private:
  // Pointer to struct gridinfo_t
  std::shared_ptr<void> _grid;
  // Pointer to SuperMatrix
  std::shared_ptr<void> _A;

  std::vector<int> cols;
  std::vector<int> rowptr;

  int m_loc, m;
  bool _verbose;
  MPI_Comm _comm;
};

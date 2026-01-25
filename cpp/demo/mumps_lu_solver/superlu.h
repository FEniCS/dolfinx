
#pragma once

#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>

template <typename T>
class SuperLUSolver
{
public:
  SuperLUSolver(MPI_Comm comm, bool verbose = false);

  ~SuperLUSolver();

  void set_operator(const dolfinx::la::MatrixCSR<T>& Amat);

  /// Solve A.u=b
  /// @param b RHS Vector
  /// @param u Solution Vector
  /// @note Must be compatible with A
  int solve(const dolfinx::la::Vector<T>& b, dolfinx::la::Vector<T>& u);

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

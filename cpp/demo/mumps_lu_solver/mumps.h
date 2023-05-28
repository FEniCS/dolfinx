
#pragma once

#include "cmumps_c.h"
#include "dmumps_c.h"
#include "smumps_c.h"
#include "zmumps_c.h"

#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>

template <typename T>
class MUMPSLUSolver
{
  // Type of C struct depending on T
  typedef typename std::conditional_t<
      std::is_same_v<T, double>, DMUMPS_STRUC_C,
      std::conditional_t<
          std::is_same_v<T, float>, SMUMPS_STRUC_C,
          std::conditional_t<
              std::is_same_v<T, std::complex<double>>, ZMUMPS_STRUC_C,
              std::conditional_t<std::is_same_v<T, std::complex<float>>,
                                 CMUMPS_STRUC_C, std::false_type>>>>
      MUMPS_STRUC_C;

  // C Scalar type depending on T
  typedef typename std::conditional_t<
      std::is_same_v<T, std::complex<double>>, mumps_double_complex,
      std::conditional_t<std::is_same_v<T, std::complex<float>>, mumps_complex,
                         T>>
      MUMPS_Type;

public:
  /// An LU Solver using MUMPS
  /// @param comm MPI_Comm
  MUMPSLUSolver(MPI_Comm comm);

  /// Destructor
  ~MUMPSLUSolver();

  /// Set the LHS matrix A
  /// @param Amat The LHS matrix to be factorised
  /// @note Analysis and factorisation takes place when calling `set_operator`
  void set_operator(const la::MatrixCSR<T>& Amat);

  /// Solve A.u=b
  /// @param b RHS Vector
  /// @param u Solution Vector
  /// @note Must be compatible with A
  int solve(const la::Vector<T>& b, la::Vector<T>& u);

private:
  // Convenient wrapper around call to MUMPS
  void mumps_c(MUMPS_STRUC_C* id);

  // Communicator
  MPI_Comm _comm;

  // MUMPS internal structure
  MUMPS_STRUC_C id;
};

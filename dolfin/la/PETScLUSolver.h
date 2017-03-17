// Copyright (C) 2005-2017 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

#ifndef __DOLFIN_PETSC_LU_SOLVER_H
#define __DOLFIN_PETSC_LU_SOLVER_H

#ifdef HAS_PETSC

#include <map>
#include <memory>
#include <string>
#include <petscmat.h>
#include <petscpc.h>
#include <dolfin/common/MPI.h>
#include "PETScKrylovSolver.h"

namespace dolfin
{
  /// Forward declarations
  class GenericLinearOperator;
  class GenericVector;
  class PETScLinearOperator;
  class PETScMatrix;
  class PETScVector;

  /// This class implements the direct solution (LU factorization) for
  /// linear systems of the form Ax = b. It is a wrapper for the LU
  /// solver of PETSc.

  class PETScLUSolver : public GenericLinearSolver
  {
  public:

    /// Constructor
    PETScLUSolver(MPI_Comm comm, std::string method="default");

    /// Constructor
    PETScLUSolver(std::string method="default");

    /// Constructor
    PETScLUSolver(MPI_Comm comm,
                  std::shared_ptr<const PETScMatrix> A,
                  std::string method="default");

    /// Constructor
    PETScLUSolver(std::shared_ptr<const PETScMatrix> A,
                  std::string method="default");

    /// Destructor
    ~PETScLUSolver();

    /// Set operator (matrix)
    void set_operator(std::shared_ptr<const GenericLinearOperator> A);

    /// Set operator (matrix)
    void set_operator(std::shared_ptr<const PETScMatrix> A);

    /// Solve linear system Ax = b
    std::size_t solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b (A^t x = b if transpose is true)
    std::size_t solve(GenericVector& x, const GenericVector& b, bool transpose);

    /// Solve linear system Ax = b
    std::size_t solve(const GenericLinearOperator& A, GenericVector& x,
                      const GenericVector& b);

    /// Solve linear system Ax = b
    std::size_t solve(const PETScMatrix& A, PETScVector& x,
                      const PETScVector& b);

    /// Sets the prefix used by PETSc when searching the options
    /// database
    void set_options_prefix(std::string options_prefix);

    /// Returns the prefix used by PETSc when searching the options
    /// database
    std::string get_options_prefix() const;

    /// Set options from the PETSc options database
    void set_from_options() const;

    /// Returns the MPI communicator
    MPI_Comm mpi_comm() const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Return PETSc KSP pointer
    KSP ksp() const;

    /// Return a list of available solver methods
    static std::map<std::string, std::string> methods();

    /// Default parameter values
    static Parameters default_parameters();

    // FIXME: These should not be friend classes
    friend class PETScSNESSolver;
    friend class PETScTAOSolver;

  private:

    // FIXME: Remove
    // Available LU solvers
    static std::map<std::string, const MatSolverPackage> lumethods;


    // Select LU solver type
    static const MatSolverPackage select_solver(MPI_Comm comm,
                                                std::string method);

    PETScKrylovSolver _solver;

  };

}

#endif

#endif

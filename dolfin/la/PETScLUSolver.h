// Copyright (C) 2005-2006 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2009-2010.
//
// First added:  2005
// Last changed: 2011-03-24

#ifndef __DOLFIN_PETSC_LU_SOLVER_H
#define __DOLFIN_PETSC_LU_SOLVER_H

#ifdef HAS_PETSC

#include <map>
#include <boost/shared_ptr.hpp>
#include <petscksp.h>
#include <petscpc.h>
#include "GenericLUSolver.h"
#include "PETScObject.h"

namespace dolfin
{
  /// Forward declarations
  class GenericMatrix;
  class GenericVector;
  class PETScKrylovMatrix;
  class PETScMatrix;
  class PETScVector;

  /// This class implements the direct solution (LU factorization) for
  /// linear systems of the form Ax = b. It is a wrapper for the LU
  /// solver of PETSc.

  class PETScLUSolver : public GenericLUSolver, public PETScObject
  {
  public:

    /// Constructor
    PETScLUSolver(std::string lu_package="default");

    /// Constructor
    PETScLUSolver(const GenericMatrix& A, std::string lu_package="default");

    /// Constructor
    PETScLUSolver(boost::shared_ptr<const PETScMatrix> A,
                  std::string lu_package="default");

    /// Destructor
    ~PETScLUSolver();

    /// Set operator (matrix)
    void set_operator(const GenericMatrix& A);

    /// Set operator (matrix)
    void set_operator(const PETScMatrix& A);

    /// Get operator (matrix)
    const GenericMatrix& get_operator() const;

    /// Solve linear system Ax = b
    uint solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b
    uint solve(const PETScMatrix& A, PETScVector& x, const PETScVector& b);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Return PETSc KSP pointer
    boost::shared_ptr<KSP> ksp() const;

    /// Default parameter values
    static Parameters default_parameters();

  private:

    // Available LU solvers
    static const std::map<std::string, const MatSolverPackage> lu_packages;

    // Select LU solver type
    const MatSolverPackage select_solver(std::string& lu_package) const;

    // Initialise solver
    void init_solver(std::string& lu_package);

    // Set PETSc operators
    void set_petsc_operators();

    // Print pre-solve report
    void pre_report(const PETScMatrix& A) const;

    /// PETSc solver pointer
    boost::shared_ptr<KSP> _ksp;

    // Operator (the matrix)
    boost::shared_ptr<const PETScMatrix> A;

  };

}

#endif

#endif

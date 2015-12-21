// Copyright (C) 2004-2015 Johan Jansson and Garth N. Wells
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
//
// Modified by Anders Logg 2005-2012
// Modified by Johan Hoffman 2005
// Modified by Andy R. Terrel 2005
// Modified by Garth N. Wells 2005-2010

#ifndef __DOLFIN_PETSC_KRYLOV_SOLVER_H
#define __DOLFIN_PETSC_KRYLOV_SOLVER_H

#ifdef HAS_PETSC

#include <map>
#include <memory>
#include <string>
#include <petscksp.h>
#include <dolfin/common/types.h>
#include "GenericLinearSolver.h"
#include "PETScObject.h"

namespace dolfin
{

  /// Forward declarations
  class GenericMatrix;
  class GenericVector;
  class PETScBaseMatrix;
  class PETScMatrix;
  class PETScVector;
  class PETScPreconditioner;
  class PETScUserPreconditioner;
  class PETScSNESSolver;
  class VectorSpaceBasis;

  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b. It is a wrapper for the Krylov solvers
  /// of PETSc.

  class PETScKrylovSolver : public GenericLinearSolver, public PETScObject
  {
  public:

    /// Create Krylov solver for a particular method and names
    /// preconditioner
    PETScKrylovSolver(std::string method = "default",
                      std::string preconditioner = "default");

    /// Create Krylov solver for a particular method and
    /// PETScPreconditioner (shared_ptr version)
    PETScKrylovSolver(std::string method,
		      std::shared_ptr<PETScPreconditioner> preconditioner);

    /// Create Krylov solver for a particular method and
    /// PETScPreconditioner (shared_ptr version)
    PETScKrylovSolver(std::string method,
                      std::shared_ptr<PETScUserPreconditioner> preconditioner);

    /// Create solver wrapper of a PETSc KSP object
    explicit PETScKrylovSolver(KSP ksp);

    /// Destructor
    ~PETScKrylovSolver();

    /// Set operator (matrix)
    void set_operator(std::shared_ptr<const GenericLinearOperator> A);

    /// Set operator (matrix) and preconditioner matrix
    void set_operators(std::shared_ptr<const GenericLinearOperator> A,
                       std::shared_ptr<const GenericLinearOperator> P);

    /// Get operator (matrix)
    const PETScBaseMatrix& get_operator() const;

    /// Solve linear system Ax = b and return number of iterations
    std::size_t solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    std::size_t solve(PETScVector& x, const PETScVector& b);

    /// Solve linear system Ax = b and return number of iterations
    std::size_t solve(const GenericLinearOperator& A, GenericVector& x,
                      const GenericVector& b);

    /// Reuse preconditioner if true, otherwise do not, even if matrix
    /// operator changes (by default preconditioner is re-built if the
    /// matrix changes)
    void set_reuse_preconditioner(bool reuse_pc);

    /// Sets the prefix used by PETSc when searching the options
    /// database
    void set_options_prefix(std::string options_prefix);

    /// Returns the prefix used by PETSc when searching the options
    /// database
    std::string get_options_prefix() const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Return PETSc KSP pointer
    KSP ksp() const;

    /// Return a list of available solver methods
    static std::map<std::string, std::string> methods();

    /// Return a list of available preconditioners
    static std::map<std::string, std::string> preconditioners();

    /// Default parameter values
    static Parameters default_parameters();

    friend class PETScSNESSolver;
    friend class PETScTAOSolver;

  private:

    // Initialize KSP solver
    void init(const std::string& method);

    // Set operator (matrix)
    void _set_operator(std::shared_ptr<const PETScBaseMatrix> A);

    // Set operator (matrix) and preconditioner matrix
    void _set_operators(std::shared_ptr<const PETScBaseMatrix> A,
                        std::shared_ptr<const PETScBaseMatrix> P);

    // Solve linear system Ax = b and return number of iterations
    std::size_t _solve(const PETScBaseMatrix& A, PETScVector& x,
                       const PETScVector& b);

    // Set options that affect KSP object
    void set_petsc_ksp_options();

    // Report the number of iterations
    void write_report(int num_iterations, KSPConvergedReason reason);

    void check_dimensions(const PETScBaseMatrix& A, const GenericVector& x,
                          const GenericVector& b) const;

    // Prefix for PETSc options database
    std::string _petsc_options_prefix;

    // Available solvers
    static const std::map<std::string, const KSPType> _methods;

    // Available solvers descriptions
    static const std::map<std::string, std::string>
      _methods_descr;

    // PETSc solver pointer
    KSP _ksp;

    // DOLFIN-defined PETScUserPreconditioner
    PETScUserPreconditioner* pc_dolfin;

    // Preconditioner
    std::shared_ptr<PETScPreconditioner> _preconditioner;

    // Operator (the matrix)
    std::shared_ptr<const PETScBaseMatrix> _matA;

    // Matrix used to construct the preconditioner
    std::shared_ptr<const PETScBaseMatrix> _matP;

    bool preconditioner_set;

  };

}

#endif

#endif

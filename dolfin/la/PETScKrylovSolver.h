// Copyright (C) 2004-2005 Johan Jansson
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
//
// First added:  2005-12-02
// Last changed: 2012-08-20

#ifndef __DOLFIN_PETSC_KRYLOV_SOLVER_H
#define __DOLFIN_PETSC_KRYLOV_SOLVER_H

#ifdef HAS_PETSC

#include <map>
#include <petscksp.h>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
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

  /// This class implements Krylov methods for linear systems
  /// of the form Ax = b. It is a wrapper for the Krylov solvers
  /// of PETSc.

  class PETScKrylovSolver : public GenericLinearSolver, public PETScObject
  {
  public:

    /// Create Krylov solver for a particular method and names preconditioner
    PETScKrylovSolver(std::string method = "default",
                      std::string preconditioner = "default");

    /// Create Krylov solver for a particular method and PETScPreconditioner
    PETScKrylovSolver(std::string method, PETScPreconditioner& preconditioner);

    /// Create Krylov solver for a particular method and PETScPreconditioner
    /// shared_ptr version
    PETScKrylovSolver(std::string method,
		      boost::shared_ptr<PETScPreconditioner> preconditioner);

    /// Create Krylov solver for a particular method and PETScPreconditioner
    PETScKrylovSolver(std::string method, PETScUserPreconditioner& preconditioner);

    /// Create Krylov solver for a particular method and PETScPreconditioner
    /// shared_ptr version
    PETScKrylovSolver(std::string method,
		      boost::shared_ptr<PETScUserPreconditioner> preconditioner);

    /// Create solver from given PETSc KSP pointer
    explicit PETScKrylovSolver(boost::shared_ptr<KSP> ksp);

    /// Destructor
    ~PETScKrylovSolver();

    /// Set operator (matrix)
    void set_operator(const boost::shared_ptr<const GenericLinearOperator> A);

    /// Set operator (matrix)
    void set_operator(const boost::shared_ptr<const PETScBaseMatrix> A);

    /// Set operator (matrix) and preconditioner matrix
    void set_operators(const boost::shared_ptr<const GenericLinearOperator> A,
                       const boost::shared_ptr<const GenericLinearOperator> P);

    /// Set operator (matrix) and preconditioner matrix
    void set_operators(const boost::shared_ptr<const PETScBaseMatrix> A,
                       const boost::shared_ptr<const PETScBaseMatrix> P);

    /// Get operator (matrix)
    const PETScBaseMatrix& get_operator() const;

    /// Solve linear system Ax = b and return number of iterations
    uint solve(GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    uint solve(PETScVector& x, const PETScVector& b);

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const GenericLinearOperator& A, GenericVector& x, const GenericVector& b);

    /// Solve linear system Ax = b and return number of iterations
    uint solve(const PETScBaseMatrix& A, PETScVector& x, const PETScVector& b);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Return PETSc KSP pointer
    boost::shared_ptr<KSP> ksp() const;

    /// Return a list of available solver methods
    static std::vector<std::pair<std::string, std::string> > methods();

    /// Return a list of available preconditioners
    static std::vector<std::pair<std::string, std::string> > preconditioners();

    /// Default parameter values
    static Parameters default_parameters();

  private:

    /// Initialize KSP solver
    void init(const std::string& method);

    // Set PETSc operators
    void set_petsc_operators();

    // Set options
    void set_petsc_options();

    /// Report the number of iterations
    void write_report(int num_iterations, KSPConvergedReason reason);

    void check_dimensions(const PETScBaseMatrix& A, const GenericVector& x,
                          const GenericVector& b) const;

    // Available solvers
    static const std::map<std::string, const KSPType> _methods;

    // Available solvers descriptions
    static const std::vector<std::pair<std::string, std::string> > _methods_descr;

    /// DOLFIN-defined PETScUserPreconditioner
    PETScUserPreconditioner* pc_dolfin;

    /// PETSc solver pointer
    boost::shared_ptr<KSP> _ksp;

    /// Preconditioner
    boost::shared_ptr<PETScPreconditioner> preconditioner;

    /// Operator (the matrix)
    boost::shared_ptr<const PETScBaseMatrix> A;

    /// Matrix used to construct the preconditoner
    boost::shared_ptr<const PETScBaseMatrix> P;

    bool preconditioner_set;

  };

}

#endif

#endif

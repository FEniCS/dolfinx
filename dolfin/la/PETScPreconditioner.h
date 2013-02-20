// Copyright (C) 2010 Garth N. Wells
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
// Modified by Anders Logg 2011
//
// First added:  2010-02-25
// Last changed: 2011-10-19

#ifndef __DOLFIN_PETSC_PRECONDITIONER_H
#define __DOLFIN_PETSC_PRECONDITIONER_H

#ifdef HAS_PETSC

#include <string>

#include <petscpc.h>
#include <boost/shared_ptr.hpp>
#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include <dolfin/la/PETScObject.h>
#include <dolfin/parameter/Parameters.h>
#include "GenericPreconditioner.h"

namespace dolfin
{

  // Forward declarations
  class PETScKrylovSolver;
  class PETScSNESSolver;


  /// This class is a wrapper for configuring PETSc preconditioners. It does
  /// not own a preconditioner. It can take a PETScKrylovSolver and set the
  /// preconditioner type and parameters.

  class PETScPreconditioner : public PETScObject, public GenericPreconditioner, public Variable
  {
  public:

    /// Create a particular preconditioner object
    explicit PETScPreconditioner(std::string type = "default");

    /// Destructor
    virtual ~PETScPreconditioner();

    /// Set the precondtioner type and parameters
    virtual void set(PETScKrylovSolver& solver);

    /// Set the (approximate) null space of the preconditioner operator
    /// (matrix). This is required for certain preconditioner types,
    /// e.g. smoothed aggregation multigrid
    void set_nullspace(const std::vector<const GenericVector*> nullspace);

    /// Return the PETSc null space
    boost::shared_ptr<const MatNullSpace> nullspace() const
    { return petsc_nullspace; }

    /// Set the coordinates of the operator (matrix) rows and geometric
    /// dimension d. This is can be used by required for certain
    /// preconditioners, e.g. ML. The input for this function can be
    /// generated using GenericDofMap::tabulate_all_dofs.
    void set_coordinates(const std::vector<double>& x, std::size_t dim);

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Rerturn a list of available preconditioners
    static std::vector<std::pair<std::string, std::string> > preconditioners();

    /// Default parameter values
    static Parameters default_parameters();

    friend class PETScSNESSolver;

  private:

    /// Named preconditioner
    std::string type;

    // Available names preconditioners
    static const std::map<std::string, const PCType> _methods;

    // Available preconditioner descriptions
    static const std::vector<std::pair<std::string, std::string> > _methods_descr;

    // Null space vectors
    std::vector<PETScVector> _nullspace;

    // PETSc null space. Would like this to be a scoped_ptr, but it
    // doesn't support custom deleters. Change to std::unique_ptr in
    // the future.
    boost::shared_ptr<MatNullSpace> petsc_nullspace;

    // Operator row coordinates
    std::vector<double> _coordinates;

    // Geometric dimension associates with coordinates
    std::size_t gdim;

  };

}

#endif

#endif

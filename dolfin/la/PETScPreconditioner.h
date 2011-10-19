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

namespace dolfin
{

  // Forward declarations
  class PETScKrylovSolver;


  /// This class is a wrapper for configuring PETSc preconditioners. It does
  /// not own a preconditioner. It can take a PETScKrylovSolver and set the
  /// preconditioner type and parameters.

  class PETScPreconditioner : public PETScObject, public Variable
  {
  public:

    /// Create a particular preconditioner object
    explicit PETScPreconditioner(std::string type = "default");

    /// Destructor
    virtual ~PETScPreconditioner();

    /// Set the precondtioner type and parameters
    virtual void set(PETScKrylovSolver& solver) const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    // Rerturn a list of available preconditioners
    static std::vector<std::pair<std::string, std::string> > preconditioners();

    /// Default parameter values
    static Parameters default_parameters();

  private:

    /// Named preconditioner
    std::string type;

    // Available names preconditioners
    static const std::map<std::string, const PCType> _methods;
  };

}

#endif

#endif

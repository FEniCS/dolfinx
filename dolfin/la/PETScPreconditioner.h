// Copyright (C) 2010 Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-25
// Last changed:

#ifndef __DOFLIN_PETSC_PRECONDITIONER_H
#define __DOFLIN_PETSC_PRECONDITIONER_H

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


  /// This class is a wrapper for configuring PETSc preconditioners

  class PETScPreconditioner : public PETScObject, public Variable
  {
  public:

    /// Create Krylov solver for a particular method and preconditioner
    explicit PETScPreconditioner(std::string type = "default");

    /// Destructor
    ~PETScPreconditioner();

    /// Set the precondtioner
    void set(PETScKrylovSolver& solver) const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Default parameter values
    static Parameters default_parameters();

  private:

    /// Named preconditioner
    std::string type;

    // Available solvers and preconditioners
    static const std::map<std::string, const PCType> methods;
  };

}

#endif

#endif

// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Last changed: 2008-05-16

#ifdef HAS_TRILINOS

#ifndef __EPETRA_PRECONDITIONER_SOLVER_H
#define __EPETRA_PRECONDITIONER_SOLVER_H

#include <dolfin/common/types.h>
#include <dolfin/parameter/Parametrized.h>
#include "enums_la.h"

#endif 

namespace dolfin 
{
  class EpetraVector; 
  class EpetraMatrix; 

  /// This class specifies the interface for user-defined Krylov
  /// method EpetraPreconditioners. A user wishing to implement her own
  /// EpetraPreconditioner needs only supply a function that approximately
  /// solves the linear system given a right-hand side.

  class EpetraPreconditioner : public Parametrized  
  {
    public: 
    /// Constructor
    EpetraPreconditioner() {};

    /// Destructor
    virtual ~EpetraPreconditioner() {};

    /// Set the Preconditioner type (amg, ilu, etc.)
    void setType(PreconditionerType type); 

    /// Initialise preconditioner 
    virtual void init(const EpetraMatrix& A);

    /// Solve linear system (M^-1)Ax = y
    virtual void solve(EpetraVector& x, const EpetraVector& b);

    private: 
    PreconditionerType prec_type; 

  }; 

}


#endif 



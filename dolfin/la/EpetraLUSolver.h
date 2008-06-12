// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Last changed: 2008-05-16

#ifdef HAS_TRILINOS
#ifndef __EPETRA_LU_SOLVER_H
#define __EPETRA_LU_SOLVER_H

#include <dolfin/parameter/Parametrized.h>
#include "EpetraVector.h"

namespace dolfin
{
  /// Forward declarations
  class EpetraMatrix;
  class EpetraKrylovMatrix;

  /// This class implements the direct solution (LU factorization) for
  /// linear systems of the form Ax = b. It is a wrapper for the LU
  /// solver of Epetra.
  
  class EpetraLUSolver : public Parametrized
  {
  public:
    
    /// Constructor
    EpetraLUSolver();

    /// Destructor
    ~EpetraLUSolver();

    /// Solve linear system Ax = b
    uint solve(const EpetraMatrix& A, EpetraVector& x, const EpetraVector& b);

    /// Display LU solver data
    void disp() const;
  };

}

#endif





#endif 



// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __LU_H
#define __LU_H

#include <petscksp.h>
#include <petscmat.h>

#include <dolfin/NewVector.h>
#include <dolfin/VirtualMatrix.h>

namespace dolfin
{

  class LU
  {
  public:
    
    /// Constructor
    LU();

    /// Destructor
    ~LU();

    /// Solve linear system Ax = b
    void solve(const VirtualMatrix& A, NewVector& x, const NewVector& b);

    /// Display LU solver data
    void disp() const;

  private:
    
    // Create dense copy of virtual matrix
    void copyToDense(const VirtualMatrix& A);

    KSP ksp;

    Mat B;
    int* idxm;
    int* idxn;

    NewVector e;
    NewVector y;

  };

}

#endif

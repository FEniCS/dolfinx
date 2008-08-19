// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-07-04
// Last changed: 2006-07-04

#ifndef __UBLAS_DUMMY_PRECONDITIONER_H
#define __UBLAS_DUMMY_PRECONDITIONER_H

#include "uBLASPreconditioner.h"

namespace dolfin
{

  /// This class provides a dummy (do nothing) preconditioner for the
  /// uBLAS Krylov solver.

  class uBLASDummyPreconditioner : public uBLASPreconditioner
  {
  public:

    /// Constructor
    uBLASDummyPreconditioner();

    /// Destructor
    ~uBLASDummyPreconditioner();

    /// Solve linear system Ax = b approximately
    void solve(uBLASVector& x, const uBLASVector& b) const;

  };

}

#endif

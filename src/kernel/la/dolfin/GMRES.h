// Copyright (C) 2004-2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-12-02
// Last changed:

#ifndef __GMRES_H
#define __GMRES_H

#include <dolfin/KrylovSolver.h>

namespace dolfin
{
  /// This class initialises the GMRES version of the Krylov solver for linear 
  /// systems.
  
  class GMRES : public KrylovSolver
  {
  public:

    /// Create GMRES solver
    GMRES();

    /// Create GMRES solver with given PETSc preconditioner
    GMRES(PreconditionerType preconditionertype);

    /// Destructor
    ~GMRES();
    
  };

}

#endif

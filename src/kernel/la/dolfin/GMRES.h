// Copyright (C) 2004-2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
// 
// First added:  2005-12-02
// Last changed: 2006-05-07

#ifndef __GMRES_H
#define __GMRES_H

#ifdef HAVE_PETSC_H

#include <dolfin/KrylovSolver.h>

namespace dolfin
{
  /// This class initialises the GMRES version of the Krylov solver for linear 
  /// systems.
  
  class GMRES : public KrylovSolver
  {
  public:

    /// Create GMRES solver with default preconditioner
    GMRES();

    /// Create GMRES solver with a particular preconditioner
    GMRES(Preconditioner::Type preconditioner);

    /// Create GMRES solver with a particular preconditioner
    GMRES(Preconditioner& preconditioner);

    /// Destructor
    ~GMRES();
    
  };

}

#endif

#endif

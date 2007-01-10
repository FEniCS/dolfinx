// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-08-15
// Last changed: 2006-08-15

#ifndef __KRYLOV_METHOD_H
#define __KRYLOV_METHOD_H

namespace dolfin
{

  /// List of predefined Krylov methods.

  enum KrylovMethod
  { 
    cg,            // Conjugate gradient method
    gmres,         // GMRES method
    bicgstab,      // Stabilised biconjugate gradient squared method 
    default_method // Default choice of Krylov method
  };

}

#endif

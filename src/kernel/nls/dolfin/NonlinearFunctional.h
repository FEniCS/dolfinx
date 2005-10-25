// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-24
// Last changed: 2005

#ifndef __NONLINEAR_FUNCTIONAL_H
#define __NONLINEAR_FUNCTIONAL_H

#include <petscsnes.h>

#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/Mesh.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>

namespace dolfin
{

  /// This class contains pointers to the necessary componets to form a 
  /// nonlinear functional F(x) and its Jacobian F'(x).
  
  class NonlinearFunctional
  {
  public:

    /// Constructor
    NonlinearFunctional();

    /// Destructor
    virtual ~NonlinearFunctional();
  
  private:

    const Mesh* mesh;
    const BilinearForm* a;
    const LinearForm* L;
    const Matrix* A;
    const Vector* b;
    const Vector* x;

  };
}

#endif

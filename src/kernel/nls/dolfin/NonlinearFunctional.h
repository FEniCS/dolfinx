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

  /// This class contains pointers to the necessary components to form a 
  /// nonlinear functional F(x) and its Jacobian F'(x).
  
  class NonlinearFunctional
  {
  public:

    /// Create nonlinear functional
    NonlinearFunctional();

    /// Create nonlinear functional with bilinear form, linear form, mesh, RHS 
    /// vector, Jacobian matrix and solution vector
    NonlinearFunctional(BilinearForm& a, LinearForm& L, Mesh& mesh, Vector& x, 
                        Matrix& A, Vector& b);

    /// Destructor
    virtual ~NonlinearFunctional();
  
    /// User-defined function to update functions in forms
    virtual void UpdateNonlinearFunction();

  private:

    const BilinearForm* _a;
    const LinearForm* _L;
    const Mesh* _mesh;
    Vector* _x0;
    Matrix* _A;
    Vector* _b;

  };
}

#endif

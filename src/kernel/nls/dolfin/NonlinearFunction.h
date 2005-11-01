// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-24
// Last changed: 2005

#ifndef __NONLINEAR_FUNCTION_H
#define __NONLINEAR_FUNCTION_H

#include <petscsnes.h>

#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/Mesh.h>
#include <dolfin/BilinearForm.h>
#include <dolfin/LinearForm.h>
#include <dolfin/BoundaryCondition.h>

namespace dolfin
{

  /// This class contains pointers to the necessary components to form the 
  /// nonlinear function F(x) and its Jacobian F'(x).
  
  class NonlinearFunction
  {
  public:

    /// Create nonlinear function
    NonlinearFunction();

    /// Create nonlinear function with bilinear form, linear form, mesh, RHS 
    /// vector, Jacobian matrix and solution vector
    NonlinearFunction(BilinearForm& a, LinearForm& L, Mesh& mesh, 
                        BoundaryCondition& bc);

    /// Destructor
    virtual ~NonlinearFunction();
  
    /// User-defined function to update functions in forms
    virtual void update(Vector& xsol);

    /// Set bilinear and linear form
    void setForm(BilinearForm &a, LinearForm&L);

    /// Return mesh
    Mesh& mesh();

    /// Return bilinear form
    BilinearForm& a();

    /// Return linear form
    LinearForm& L();

    /// Return RHS vector
    BoundaryCondition& bc();

  friend class NonlinearSolver;

  private:

    BilinearForm* _a;
    LinearForm* _L;
    Mesh* _mesh;
    BoundaryCondition* _bc;

    Matrix* _A;
    Vector* _b;

  };
}

#endif

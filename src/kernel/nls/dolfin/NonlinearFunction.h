// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-10-24
// Last changed: 2005

#ifndef __NONLINEAR_FUNCTION_H
#define __NONLINEAR_FUNCTION_H

#include <petscsnes.h>

#include <dolfin/constants.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>

namespace dolfin
{

  /// This class contains pointers to the nonlinear function F(u) and its 
  /// Jacobian J = dF(u)/du.
  
  class NonlinearFunction
  {
  public:

    /// Create nonlinear function
    NonlinearFunction();

    /// Destructor
    virtual ~NonlinearFunction();
  
    /// Set pointer to F
    void setF(Vector& b);

    /// Set pointer to Jacobian matrix
    void setJ(Matrix& A); 

     /// User-defined function to compute system dimension
    virtual uint size();

     /// User-defined function to compute F(u)
    virtual void F(Vector& b, const Vector& x);

     /// User-defined function to compute Jacobian
    virtual void J(Matrix& A, const Vector& x);

     /// User-defined function to compute F(u) and Jacobian
    virtual void form(Matrix& A, Vector& b, const Vector& x);

    /// Return Jacobian
    Matrix& J() const;

    /// Return RHS vector
    Vector& F() const;

  private:

    Matrix* _A;
    Vector* _b;

  };
}

#endif

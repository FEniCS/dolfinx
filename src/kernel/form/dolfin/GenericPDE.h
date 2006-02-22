// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-21
// Last changed: 

#ifndef __GENERIC_PDE_H
#define __GENERIC_PDE_H

#include <dolfin/Parametrized.h>

namespace dolfin
{
  
  class BilinearForm;
  class BoundaryCondition;
  class Function;
  class LinearForm;
  class Matrix;
  class Mesh;
  class Vector;

  /// This class serves as a base class/interface for specific PDE's.

  class GenericPDE : public Parametrized
  {
  public:

    /// Constructor
    GenericPDE();

    /// Destructor
    virtual ~GenericPDE();

//     /// Compute RHS vector and (approximate) Jacobian matrix for PDE
//    virtual void form(Matrix& A, Vector& b, const Vector& x) = 0;

//     /// Compute RHS vector for PDE
//    virtual void F(Vector& b, const Vector& x) = 0;

//     /// Compute (approximate) Jacobian/stiffness matrix for PDE
//    virtual void J(Matrix& A, const Vector& x) = 0;

     /// Solve
    virtual Function solve() = 0;

     /// Solve
    virtual void solve(Function& u) = 0;

     /// Solve
    virtual void solve(Function& u0, Function& u1) = 0;

     /// Solve
    virtual void solve(Function& u0, Function& u1, Function& u2) = 0;

     /// Solve
    virtual void solve(Function& u0, Function& u1, Function& u2, Function& u3) = 0;

    /// Return the bilinear form mesh associated with PDE (if any)
    virtual BilinearForm& a() = 0;

    /// Return the linear form mesh associated with PDE (if any)
    virtual LinearForm& L() =0;

    /// Return the mesh associated with PDE (if any)
    virtual Mesh& mesh() = 0;

    /// Return the boundary conditions associated with PDE (if any)
    virtual BoundaryCondition& bc() = 0;

  };

}

#endif

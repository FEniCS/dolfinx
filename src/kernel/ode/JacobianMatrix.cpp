// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ODE.h>
#include <dolfin/JacobianMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
JacobianMatrix::JacobianMatrix(ODE& ode) :
  Matrix(ode.size(), ode.size(), Matrix::generic), dfdu(ode.size(), ode.size())
{
  // Here we should create a sparse matrix with all zeros that
  // has the same sparsity pattern as the ode (use ODE::sparsity)

}
//-----------------------------------------------------------------------------
JacobianMatrix::~JacobianMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void JacobianMatrix::update(real t)
{
  // Recompute the Jacobian of the right-hand side

  // There should be a virtual method dfdu(i, j) in the ODE class that
  // people can implement if they want to provide the Jacobian explicitly,
  // instead of using a numerically computed Jacobian

}
//-----------------------------------------------------------------------------
void JacobianMatrix::mult(const Vector& x, Vector& Ax) const
{
  // Perform multiplication with vector, using the Jacobian dfdu of the
  // right-hand side f and the structure of the time slab

  // Note that we should use the new interface to the GMRES solver:
  // GMRES::solve(A, x, b)

  // Perform the multiplication by iterating over all elements in the slab.
  // For each element, check the component index of that element to pick
  // the correct elements from the Jacobian dfdu. Update a counter while
  // iterating through all elements to compute the correct value Ax(i) for
  // each degree of freedom. We also need to know if an element is the first
  // of that component in the slab, or if it has a predecessor. If it has a
  // predecessor, we need to include the derivative -1 with respect to the
  // end time value of the previous element.
  
  

}
//-----------------------------------------------------------------------------

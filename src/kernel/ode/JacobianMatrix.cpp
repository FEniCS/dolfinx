// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/NewArray.h>
#include <dolfin/RHS.h>
#include <dolfin/ODE.h>
#include <dolfin/JacobianMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
JacobianMatrix::JacobianMatrix(RHS& f) :
  Matrix(0, 0, Matrix::generic), f(f), dfdu(f.size(), f.size()), n(0)
{

  // Here we should create a sparse matrix with all zeros that
  // has the same sparsity pattern as the ode (use ODE::sparsity)

  cout << "Creating Jacobian matrix" << endl;

  for (unsigned int i = 0; i < dfdu.size(0); i++)
  {
    // Get dependencies
    NewArray<unsigned int>& row = f.ode.sparsity.row(i);

    // Copy sparsity pattern to matrix
    dfdu.initrow(i, row.size());
    for (unsigned int pos = 0; pos < row.size(); pos++)
    {
      dfdu(i, row[pos]) = 0.0;
    }
  }
}
//-----------------------------------------------------------------------------
JacobianMatrix::~JacobianMatrix()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
unsigned int JacobianMatrix::size(unsigned int dim) const
{
  return n;
}
//-----------------------------------------------------------------------------
void JacobianMatrix::update(real t, unsigned int n)
{
  // Recompute the Jacobian of the right-hand side at given time

  cout << "Updating Jacobian at time t = " << t << endl;

  // Update the right-hand side to given time
  f.update(t);

  // Compute the Jacobian
  for (unsigned int i = 0; i < dfdu.size(); i++)
  {
     unsigned tmp = 0;
     const NewArray<unsigned int>& row = f.ode.sparsity.row(i);
     for (unsigned int pos = 0; pos < row.size(); ++pos)
       dfdu(i, tmp, pos) = f.dfdu(i, row[pos]);
  }

  // Update number of unkowns
  this->n = n;
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

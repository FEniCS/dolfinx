// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/NewArray.h>
#include <dolfin/RHS.h>
#include <dolfin/ODE.h>
#include <dolfin/Element.h>
#include <dolfin/ElementIterator.h>
#include <dolfin/ElementGroupList.h>
#include <dolfin/cGqElement.h>
#include <dolfin/dGqElement.h>
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
    const NewArray<unsigned int>& row = f.ode.sparsity.row(i);

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
  // Perform the multiplication by iterating over all elements in the slab.
  // For each element, check the component index of that element to pick
  // the correct elements from the Jacobian dfdu. Update a counter while
  // iterating through all elements to compute the correct value Ax(i) for
  // each degree of freedom. We also need to know if an element is the first
  // of that component in the slab, or if it has a predecessor. If it has a
  // predecessor, we need to include the derivative -1 with respect to the
  // end time value of the previous element.
  
  /*
  unsigned int dof = 0;
  for (ElementIterator element(list); !element.end(); ++element)
  { 
    // Check element type
    if ( element->type() == Element::cg )
      dof = cGmult(x, Ax, dof, *element);
    else
      dof = dGmult(x, Ax, dof, *element);
  }
  */
}
//-----------------------------------------------------------------------------
unsigned int JacobianMatrix::cGmult(const Vector& x, Vector& Ax,
				    unsigned int dof, 
				    const cGqElement& element)
{
  /*

  // Get the component index
  unsigned int i = element.index();

  // Get the degree
  unsigned int q = element.order();

  // Iterate over local degrees of freedom
  for (unsigned int m = 0; m < q; m++)
  {
    // Global number of current degree of freedom
    const unsigned int current = dof + m;

    // Derivative w.r.t. the current degree of freedom
    Ax(current) = x(current);

    // Derivative w.r.t. the end-time value of the previous element
    if ( latest[i] != -1 )
      Ax(current) -= x(latest[i]);
    
    // Iterate over dependencies
    unsigned int j;
    for (unsigned int pos = 0; pos < row.size(); ++pos)
    {
      // Get derivative
      dfiduj = dfdu(i, j, pos);

      // Check if the component depends on itself
      if ( i == j )
      {
	// Derivative w.r.t. internal degrees of freedom on the element
	for (unsigned int n = 0; n < q; n++)
	{
	  
	  
	}
      }
      else
      {
	// Derivative w.r.t. the latest degree of freedom for other component



      }
    }
  }

  // Increase the degree of freedom
  return dof + q;

  */

  return 0;
}
//-----------------------------------------------------------------------------
unsigned int JacobianMatrix::dGmult(const Vector& x, Vector& Ax,
				    unsigned int dof,
				    const dGqElement& element)
{
  //return dof + q + 1;
  return 0;
}
//-----------------------------------------------------------------------------

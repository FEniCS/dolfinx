// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/NewArray.h>
#include <dolfin/RHS.h>
#include <dolfin/ODE.h>
#include <dolfin/Element.h>
#include <dolfin/ElementIterator.h>
#include <dolfin/ElementGroupList.h>
#include <dolfin/cGqMethods.h>
#include <dolfin/dGqMethods.h>
#include <dolfin/JacobianMatrix.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
JacobianMatrix::JacobianMatrix(RHS& f) : f(f){
dolfin_error("This function needs to be updated to the new format.");
/*

:
  Matrix(0, 0, Matrix::generic),
  f(f), dfdu(f.size(), f.size()), n(0), elements(0), latest(f.size()), t0(0)
{

  // Here we should create a sparse matrix with all zeros that
  // has the same sparsity pattern as the ode (use ODE::sparsity)

  cout << "Creating Jacobian matrix" << endl;
  cout << "Size of system: " << f.size() << endl;

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

    // Set the latest known index to 0 for each component, will be changed
    // to the correct value by the update() function and then updated
    // throughout the multiplication.
    latest[i] = 0;
  }
*/
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
void JacobianMatrix::update(real t)
{
dolfin_error("This function needs to be updated to the new format.");
/*
  // Update the right-hand side to given time
  f.update(t);

  // Compute the Jacobian
  for (unsigned int i = 0; i < dfdu.size(0); i++)
  {
     unsigned tmp = 0;
     const NewArray<unsigned int>& row = f.ode.sparsity.row(i);
     for (unsigned int pos = 0; pos < row.size(); ++pos)
       dfdu(i, tmp, pos) = f.dfdu(i, row[pos]);
  }
*/
}
//-----------------------------------------------------------------------------
unsigned int JacobianMatrix::update(ElementGroupList& elements)
{
  // Update time slab                                                           
  this->elements = &elements;

  // Update the start time for the time slab                                    
  ElementIterator element(elements);
  t0 = element->starttime();

  // Compute the number of unknowns and the latest indices.                     
  n = 0;
  for (ElementIterator element(elements); !element.end(); ++element)
  {
    n += element->size();
  }

  predecessors.resize(n);

  int i = 0;
  for (ElementIterator element(elements); !element.end(); ++element)
  {
    if(element->starttime() > t0)
    {
      predecessors[i] = latest[element->index()];
    }
    latest[element->index()] = i;
    i += element->size();
  }

  // Return the number of unknowns
  return n;
}
//-----------------------------------------------------------------------------
void JacobianMatrix::show() const
{
dolfin_error("This function needs to be updated to the new format.");
/*
  // Since we don't really have the matrix, we create the matrix by
  // performing multiplication with unit vectors. Used only for debugging.

  Matrix A(n, n, Matrix::dense);
  Vector x(n), y(n);
  x = 0.0;
  
  for (unsigned int j = 0; j < n; j++)
  {
    x(j) = 1.0;
    x.show();
    mult(x, y);
    for (unsigned int i = 0; i < n; i++)
      A(i, j) = y(i);
    x(j) = 0.0;
  }

  A.show();
*/
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
  
  // Check that we have the time slab
  if ( !elements )
    dolfin_error("Elements of the time slab not supplied.");
  
  // Reset the output vector
  Ax = 0.0;

  unsigned int dof = 0;
  for (ElementIterator element(*elements); !element.end(); ++element)
  { 
    // Check element type
    if ( element->type() == dolfin::Element::cg )
      dof = cGmult(x, Ax, dof, *element);
    else
       dof = dGmult(x, Ax, dof, *element);
  }
}
//-----------------------------------------------------------------------------
unsigned int JacobianMatrix::cGmult(const Vector& x, Vector& Ax,
				    unsigned int dof, 
				    const dolfin::Element& element) const
{
dolfin_error("This function needs to be updated to the new format.");
  /*
  // Get the component index
  const unsigned int i = element.index();

  // Get the degree
  const unsigned int q = element.order();

  // Get the time step
  const real k = element.timestep();

  // Check if the element has a predecessor within the slab
  const bool predecessor = element.starttime() > t0;

  // Iterate over local degrees of freedom
  for (unsigned int m = 0; m < q; m++)
  {
    // Global number of current degree of freedom
    const unsigned int current = dof + m;

    // Derivative w.r.t. the current degree of freedom
    real y = x(current);

    // Derivative w.r.t. the end-time value of the previous element
    if ( predecessor )
    {
      //y -= x(latest[i]);
      y -= x(predecessors[current]);
    }

    // Iterate over dependencies
    unsigned int j;
    for (unsigned int pos = 0; !dfdu.endrow(i, pos); ++pos)
    {
      // Get derivative
      const real df = k*dfdu(i, j, pos);

      // Check if the component depends on itself
      if ( i == j )
      {
	// Derivative w.r.t. internal degrees of freedom on the element
	for (unsigned int n = 0; n < q; n++)
	  y -= df*cG(q).weight(m + 1, n + 1)*x(dof + n);
      }
      else
      {
	// Derivative w.r.t. the latest degree of freedom for other component
	y -= df*cG(q).weightsum(m + 1)*x(latest[j]);
      }
    }

    // Update value of product
    Ax(current) = y;
  }

  // Increase the degree of freedom
  return dof + q;
  */
 return 0; // Take out!!
}
//-----------------------------------------------------------------------------
unsigned int JacobianMatrix::dGmult(const Vector& x, Vector& Ax,
				    unsigned int dof,
				    const dolfin::Element& element) const
{
dolfin_error("This function needs to be updated to the new format.");

/*
  // Get the component index
  const unsigned int i = element.index();

  // Get the degree
  const unsigned int q = element.order();

  // Get the time step
  const real k = element.timestep();

  // Check if the element has a predecessor within the slab
  const bool predecessor = element.starttime() > t0;

  // Iterate over local degrees of freedom
  for (unsigned int m = 0; m <= q; m++)
  {
    // Global number of current degree of freedom
    const unsigned int current = dof + m;

    // Derivative w.r.t. the current degree of freedom
    real y = x(current);

    // Derivative w.r.t. the end-time value of the previous element
    if ( predecessor )
    {
      y -= x(predecessors[current]);
      //y -= x(latest[i]);
    }    

    // Iterate over dependencies
    unsigned int j;
    for (unsigned int pos = 0; !dfdu.endrow(i, pos); ++pos)
    {
      // Get derivative
      const real df = k*dfdu(i, j, pos);

      // Check if the component depends on itself
      if ( i == j )
      {
	// Derivative w.r.t. internal degrees of freedom on the element
	for (unsigned int n = 0; n <= q; n++)
	  y -= df*dG(q).weight(m, n)*x(dof + n);
      }
      else
      {
	// Derivative w.r.t. the latest degree of freedom for other component
	y -= df*dG(q).weightsum(m)*x(latest[j]);
      }
    }

    // Update value of product
    Ax(current) = y;
  }

  // Increase the degree of freedom
  return dof + q + 1;
*/
 return 0; // Take out!!
}
//-----------------------------------------------------------------------------

// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Jansson, 2004.

#include <dolfin/dolfin_log.h>
#include <dolfin/RHS.h>
#include <dolfin/NewArray.h>
#include <dolfin/Vector.h>
#include <dolfin/Element.h>

using namespace dolfin;

// Initialize static data
Vector dolfin::Element::f;
NewArray<Matrix> dolfin::Element::A;
NewArray<Vector> dolfin::Element::b;
NewArray<Vector> dolfin::Element::x;

//-----------------------------------------------------------------------------
Element::Element(unsigned int q, unsigned int index, real t0, real t1) :
  q(q), _index(index), t0(t0), t1(t1)
{
  dolfin_error("This function needs to be updated to the new format.");

  /*
  dolfin_assert(t1 > t0);
  
  // Allocate the list of nodal values
  values = new real[q+1];

  for (unsigned int i = 0; i < q+1; i++)
    values[i] = 0.0;

  // Increase size of the common vector to the maximum required size
  // among all elements. The vector is short (of length max(q+1)), but
  // a lot of extra storage would be required if each element stored
  // its own vector.
  
  if ( (q+1) > f.size() )
    f.init(q+1);

  // Initialize the local element matrix if needed.
  // For cG(q) elements, we use matrix number q-1, which is of size q x q.
  // For dG(q) elements, we use matrix number q, which is of size (q+1) x (q+1).
  // We thus allocate matrices for of size (n+1) x (n+1) for n = 0,...,q,
  // which is enough for both cG(q) and dG(q). One more than necessary for
  // cG(q), but this is done only once.

  if ( (q+1) > A.size() )
  {
    for (unsigned int n = A.size(); n <= q; n++)
    {
      // Create matrix
      Matrix AA(n+1, n+1, Matrix::dense);
      A.push_back(AA);

      // Create vector x
      Vector xx(n+1);
      x.push_back(xx);

      // Create vector b
      Vector bb(n+1);
      b.push_back(xx);
    }
  }
  */
}
//-----------------------------------------------------------------------------
Element::~Element()
{
  // Delete values
  if ( values )
    delete [] values;
  values = 0;
}
//-----------------------------------------------------------------------------
unsigned int Element::order() const
{
  return q;
}
//-----------------------------------------------------------------------------
real Element::value(unsigned int node) const
{
  dolfin_assert(node <= q);  
  return values[node];
}
//-----------------------------------------------------------------------------
real Element::endval() const
{
  return values[q];
}
//-----------------------------------------------------------------------------
void Element::update(unsigned int node, real value)
{
  dolfin_assert(node <= q);
  values[node] = value;
}
//-----------------------------------------------------------------------------
unsigned int Element::index() const
{
  return _index;
}
//-----------------------------------------------------------------------------
real Element::starttime() const
{
  return t0;
}
//-----------------------------------------------------------------------------
real Element::endtime() const
{
  return t1;
}
//-----------------------------------------------------------------------------
real Element::computeResidual(RHS& f)
{
  // FIXME: Include jumps for dG(q)
  return ddx() - f(_index, q, t1);
}
//-----------------------------------------------------------------------------

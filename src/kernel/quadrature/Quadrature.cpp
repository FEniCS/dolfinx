#include <dolfin/Quadrature.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Quadrature::Quadrature(int n)
{
  this->n = n;
  
  p = new Point[n];
  w = new real[n];

  for (int i = 0; i < 0; i++)
	 w[i] = 0;
}
//-----------------------------------------------------------------------------
Quadrature::~Quadrature()
{
  if ( p )
	 delete [] p;
  p = 0;

  if ( w )
	 delete [] w;
  w = 0;
}
//-----------------------------------------------------------------------------
int Quadrature::size() const
{
  return n;
}
//-----------------------------------------------------------------------------
Point Quadrature::point(int i) const
{
  return p[i];
}
//-----------------------------------------------------------------------------
real Quadrature::weight(int i) const
{
  return w[i];
}
//-----------------------------------------------------------------------------
real Quadrature::measure() const
{
  return m;
}
//-----------------------------------------------------------------------------

// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_settings.h>
#include <dolfin/Point.h>
#include <dolfin/Cell.h>
#include <dolfin/Mesh.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/GenericFunction.h>
#include <dolfin/DofFunction.h>
#include <dolfin/ExpressionFunction.h>
#include <dolfin/ScalarExpressionFunction.h>
#include <dolfin/VectorExpressionFunction.h>
#include <dolfin/ODEFunction.h>
#include <dolfin/Function.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Function::Function()
{
  f = new GenericFunction();
  rename("u", "An unspecified function");
}
//-----------------------------------------------------------------------------
Function::Function(Mesh& mesh, dolfin::Vector& x, int dim, int size)
{
  f = 0;
  rename("u", "A function");
  init(mesh, x, dim, size);
}
//-----------------------------------------------------------------------------
Function::Function(const char *name, int dim, int size)
{
  f = 0;
  rename("u", "A function");
  init(name, dim, size);
}
//-----------------------------------------------------------------------------
Function::Function(unsigned int N)
{
  f = 0;
  rename("u", "A function");
  init(N);
}
//-----------------------------------------------------------------------------
Function::~Function()
{
  if ( f != 0 )
    delete f;
  f = 0;
}
//-----------------------------------------------------------------------------
void Function::init(Mesh& mesh, dolfin::Vector& x, int dim, int size)
{
  if ( f )
    delete f;
  
  f = new DofFunction(mesh, x, dim, size);
}
//-----------------------------------------------------------------------------
void Function::init(const char* name,  int dim, int size)
{
  if ( f )
    delete f;

  function fp;
  vfunction vfp;
  
  if ( size == 1 )
  {
    fp = dolfin_get(name);
    f = new ScalarExpressionFunction(fp);
  }
  else
  {
    vfp = dolfin_get(name);
    f = new VectorExpressionFunction(vfp, dim, size);
  }
}
//-----------------------------------------------------------------------------
void Function::init(unsigned int N)
{
  if ( f )
    delete f;

  f = new ODEFunction(N);
}
//-----------------------------------------------------------------------------
real Function::operator() (const Node& n, real t) const
{
  return (*f)(n, t);
}
//-----------------------------------------------------------------------------
real Function::operator() (const Node& n, real t)
{
  return (*f)(n, t);
}
//-----------------------------------------------------------------------------
real Function::operator() (const Point& p, real t) const
{
  return (*f)(p, t);
}
//-----------------------------------------------------------------------------
real Function::operator() (const Point& p, real t)
{
  return (*f)(p, t);
}
//-----------------------------------------------------------------------------
real Function::operator() (real x, real y, real z, real t) const
{
  return (*f)(x, y, z, t);
}
//-----------------------------------------------------------------------------
real Function::operator() (real x, real y, real z, real t)
{
  return (*f)(x, y, z, t);
}
//-----------------------------------------------------------------------------
real Function::operator() (unsigned int i, real t) const
{
  return (*f)(i, t);
}
//-----------------------------------------------------------------------------
real Function::operator() (unsigned int i, real t)
{
  return (*f)(i, t);
}
//-----------------------------------------------------------------------------
void Function::update(real t)
{
  f->update(t);
}
//-----------------------------------------------------------------------------
real Function::time() const
{
  return f->time();
}
//-----------------------------------------------------------------------------
Mesh& Function::mesh() const
{
  return f->mesh();
}
//-----------------------------------------------------------------------------
ElementData& Function::elmdata()
{
  return f->elmdata();
}
//-----------------------------------------------------------------------------
void Function::update(FunctionSpace::ElementFunction& v,
		      const FiniteElement& element,
		      const Cell& cell, real t) const
{
  // Update degrees of freedom for element function, assuming it belongs to
  // the local trial space of the finite element.

  // Set dimension of function space for element function
  v.init(element.dim());
   
  // Update coefficients
  f->update(v, element, cell, t);
}
//-----------------------------------------------------------------------------
// Vector function
//-----------------------------------------------------------------------------
Function::Vector::Vector(Mesh& mesh, dolfin::Vector& x, int size)
{
  f = new (Function *)[size];
  
  for (int i = 0; i < size; i++)
    f[i] = new Function(mesh, x, i, size);
  
  _size = size;
}
//-----------------------------------------------------------------------------
Function::Vector::Vector(const char* name, int size)
{
  f = new (Function *)[size];

  for (int i = 0; i < size; i++)
    f[i] = new Function(name, i, size);
  
  _size = size;
}
//-----------------------------------------------------------------------------
Function::Vector::~Vector()
{
  for (int i = 0; i < _size; i++)
    delete f[i];
  delete [] f;
}
//-----------------------------------------------------------------------------
Function& Function::Vector::operator() (int i)
{
  return *f[i];
}
//-----------------------------------------------------------------------------
int Function::Vector::size() const
{
  return _size;
}
//-----------------------------------------------------------------------------

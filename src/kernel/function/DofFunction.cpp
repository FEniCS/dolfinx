// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/Vector.h>
#include <dolfin/ElementFunction.h>
#include <dolfin/FiniteElement.h>
#include <dolfin/NewFiniteElement.h>
#include <dolfin/Node.h>
#include <dolfin/Point.h>
#include <dolfin/NewPDE.h>
#include <dolfin/DofFunction.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
DofFunction::DofFunction(Mesh& mesh, Vector& x, int dim, int size) :
  GenericFunction(), _mesh(mesh), x(x), t(0), dim(dim), size(size)
{
  // FIXME: assumes nodal basis
  x.init(mesh.noNodes() * size);
}
//-----------------------------------------------------------------------------
DofFunction::~DofFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real DofFunction::operator() (const Node& n, real t)  const
{
  return x(n.id()*size + dim);
}
//-----------------------------------------------------------------------------
real DofFunction::operator() (const Node& n, real t)
{
  return x(n.id()*size + dim);
}
//-----------------------------------------------------------------------------
void DofFunction::update(real t)
{
  this->t = t;
}
//-----------------------------------------------------------------------------
real DofFunction::time() const
{
  return t;
}
//-----------------------------------------------------------------------------
Mesh& DofFunction::mesh() const
{
  return _mesh;
}
//-----------------------------------------------------------------------------
void DofFunction::update(FunctionSpace::ElementFunction &v,
			 const FiniteElement &element,
			 const Cell &cell,
			 real t) const
{
  for (FiniteElement::TrialFunctionIterator phi(element); !phi.end(); ++phi)
    v.set(phi.index(), phi, x(phi.dof(cell) * size + dim));
}
//-----------------------------------------------------------------------------
void DofFunction::update(NewArray<real>& w, const Cell& cell, 
			 const NewFiniteElement& element) const
{
  // FIXME: A finite element containing the mapping from local to global
  // FIXME: degrees of freedom should be part of a Function. Temporarily,
  // FIXME: we obtain the element as an argument

  w.resize(element.spacedim());
  for (unsigned int i = 0; i < element.spacedim(); i++)
    w[i] = x(element.dof(i, cell));
}
//-----------------------------------------------------------------------------

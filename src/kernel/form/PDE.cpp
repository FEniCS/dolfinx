// Copyright (C) 2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006
//
// First added:  2006-02-21
// Last changed: 2006-02-22

#include <dolfin/Function.h>
#include <dolfin/PDE.h>
#include <dolfin/LinearPDE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PDE::PDE(BilinearForm& a, LinearForm& L, Mesh& mesh)
  : Parametrized(), pde(0), _type(linear)
{
  pde = new LinearPDE(a, L, mesh);
  pde->set("parent", *this);
}
//-----------------------------------------------------------------------------
PDE::PDE(BilinearForm& a, LinearForm& L, Mesh& mesh, BoundaryCondition& bc)
  : Parametrized(),pde(0),  _type(linear)
{
  pde = new LinearPDE(a, L, mesh, bc);
  pde->set("parent", *this);
}
//-----------------------------------------------------------------------------
PDE::~PDE()
{
  delete pde;
}
//-----------------------------------------------------------------------------
Function PDE::solve()
{
  Function u;
  solve(u);
  return u;
}
//-----------------------------------------------------------------------------
void PDE::solve(Function& u)
{
  pde->solve(u);
}
//-----------------------------------------------------------------------------
void PDE::solve(Function& u0, Function& u1)
{
  pde->solve(u0, u1);
}
//-----------------------------------------------------------------------------
void PDE::solve(Function& u0, Function& u1, Function& u2)
{
  pde->solve(u0, u1, u2);
}
//-----------------------------------------------------------------------------
void PDE::solve(Function& u0, Function& u1, Function& u2, Function& u3)
{
  pde->solve(u0, u1, u2, u3);
}
//-----------------------------------------------------------------------------
BilinearForm& PDE::a()
{
  return pde->a();
}
//-----------------------------------------------------------------------------
LinearForm& PDE::L()
{
  return pde->L();
}
//-----------------------------------------------------------------------------
Mesh& PDE::mesh()
{
  return pde->mesh();
}
//-----------------------------------------------------------------------------
PDE::Type PDE::type() const
{
  return _type;
}
//-----------------------------------------------------------------------------

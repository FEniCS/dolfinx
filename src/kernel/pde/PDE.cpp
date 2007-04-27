// Copyright (C) 2007 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-21
// Last changed: 2007-04-17

#include <dolfin/dolfin_log.h>
#include <dolfin/Form.h>
#include <dolfin/Function.h>
#include <dolfin/GenericPDE.h>
#include <dolfin/LinearPDE.h>
#include <dolfin/NonlinearPDE.h>
#include <dolfin/PDE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PDE::PDE(Form& a, Form& L, Mesh& mesh, Type type)
  : pde(0), _type(type)
{
  // Create empty array of boundary conditions
  Array<BoundaryCondition*> bcs;

  // Initialize
  init(a, L, mesh, bcs);
}
//-----------------------------------------------------------------------------
PDE::PDE(Form& a, Form& L, Mesh& mesh, BoundaryCondition& bc, Type type)
  : pde(0), _type(type)
{
  // Create array with one boundary condition
  Array<BoundaryCondition*> bcs;
  bcs.push_back(&bc);

  // Initialize
  init(a, L, mesh, bcs);
}
//-----------------------------------------------------------------------------
PDE::PDE(Form& a, Form& L, Mesh& mesh, Array<BoundaryCondition*>& bcs, Type type)
  : pde(0), _type(type)
{
  // Initialize
  init(a, L, mesh, bcs);
}
//-----------------------------------------------------------------------------
PDE::~PDE()
{
  if ( pde )
    delete pde;
}
//-----------------------------------------------------------------------------
void PDE::solve(Function& u)
{
  pde->solve(u);
}
//-----------------------------------------------------------------------------
PDE::Type PDE::type() const
{
  return _type;
}
//-----------------------------------------------------------------------------
void PDE::init(Form& a, Form& L, Mesh& mesh, Array<BoundaryCondition*>& bcs)
{
  // Initialize PDE instance
  switch ( _type )
  {
  case linear:
    pde = new LinearPDE(a, L, mesh, bcs);
    break;
  case nonlinear:
    pde = new NonlinearPDE(a, L, mesh, bcs);
    break;
  default:
    dolfin_error("Unknown PDE type");
  }

  // Parametrize PDE instance
  pde->set("parent", *this);
}
//-----------------------------------------------------------------------------

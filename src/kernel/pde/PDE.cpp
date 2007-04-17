// Copyright (C) 2007 Anders Logg and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
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
  switch ( type )
  {
  case linear:
    pde = new LinearPDE(a, L, mesh);
    break;
  case nonlinear:
    pde = new NonlinearPDE(a, L, mesh);
    break;
  default:
    dolfin_error("Unknown PDE type");
  }

  pde->set("parent", *this);
}
//-----------------------------------------------------------------------------
PDE(Form& a, Form& L, Mesh& mesh, BoundaryCondition& bc, Type type)
  : pde(0), _type(type)
{
  switch ( type )
  {
  case linear:
    pde = new LinearPDE(a, L, mesh, bc);
    break;
  case nonlinear:
    pde = new NonlinearPDE(a, L, mesh, bc);
    break;
  default:
    dolfin_error("Unknown PDE type");
  }

  pde->set("parent", *this);
}
//-----------------------------------------------------------------------------


*/
}
//-----------------------------------------------------------------------------
PDE::PDE(BilinearForm& a, LinearForm& L, Mesh& mesh, BoundaryCondition& bc, 
    Type pde_type) : Parametrized(), pde(0), _type(pde_type)
{
  dolfin_error("PDE has not yet been updated for new UFC structure.");
/*
  switch(pde_type)
  {
  case linear:
    pde = new LinearPDE(a, L, mesh, bc);
    break;
  case nonlinear:
    pde = new NonlinearPDE(a, L, mesh, bc);
    break;

  
  pde->set("parent", *this);
*/
}
//-----------------------------------------------------------------------------
PDE::~PDE()
{
  delete pde;
}
//-----------------------------------------------------------------------------
Function PDE::solve()
{
  // FIXME: Temporary fix
  UnitSquare mesh(1, 1);
  
  Function u(mesh);
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
  dolfin_error("PDE has not yet been updated for new UFC structure.");
/*
  // Check size of mixed system
  if ( pde->elementdim() != 2 )
  {
    dolfin_error1("Size of mixed system (%d) does not match number of functions (2).",
      pde->elementdim());
  }

  // Solve mixed system
  Function u;
  solve(u);

  /// Extract sub functions
  dolfin_info("Extracting sub functions from mixed system.");
  u0 = u[0];
  u1 = u[1];
*/
}
//-----------------------------------------------------------------------------
void PDE::solve(Function& u0, Function& u1, Function& u2)
{
  dolfin_error("PDE has not yet been updated for new UFC structure.");
/*
  // Check size of mixed system
  if ( pde->elementdim() != 3 )
  {
    dolfin_error1("Size of mixed system (%d) does not match number of functions (3).",
      pde->elementdim());
  }

  // Solve mixed system
  Function u;
  solve(u);

  /// Extract sub functions
  dolfin_info("Extracting sub functions from mixed system.");
  u0 = u[0];
  u1 = u[1];
  u2 = u[2];
*/
}
//-----------------------------------------------------------------------------
void PDE::solve(Function& u0, Function& u1, Function& u2, Function& u3)
{
  dolfin_error("PDE has not yet been updated for new UFC structure.");
/*
  // Check size of mixed system
  if ( pde->elementdim() != 4 )
  {
    dolfin_error1("Size of mixed system (%d) does not match number of functions (4).",
      pde->elementdim());
  }

  // Solve mixed system
  Function u;
  solve(u);

  /// Extract sub functions
  dolfin_info("Extracting sub functions from mixed system.");
  u0 = u[0];
  u1 = u[1];
  u2 = u[2];
  u3 = u[3];
*/
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

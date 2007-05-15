// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2006-2007.
//
// First added:  2005-10-24
// Last changed: 2007-05-15

#include <dolfin/BoundaryCondition.h>
#include <dolfin/Function.h>
#include <dolfin/NonlinearPDE.h>
#include <dolfin/Form.h>
#include <dolfin/dolfin_log.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NonlinearPDE::NonlinearPDE(Form& a,
                           Form& L,
                           Mesh& mesh,
                           BoundaryCondition& bc)
  : a(a), L(L), mesh(mesh), bcs(bcs)
{
  message("Creating nonlinear PDE with %d boundary condition(s).", bcs.size());

  // Check ranks of forms
  if ( a.form().rank() != 2 )
    error("Expected a bilinear form but rank is %d.", a.form().rank());
  if ( L.form().rank() != 1 )
    error("Expected a linear form but rank is %d.", L.form().rank());

  // Create array with one boundary condition
  bcs.push_back(&bc);
}
//-----------------------------------------------------------------------------
NonlinearPDE::NonlinearPDE(Form& a,
                           Form& L,
                           Mesh& mesh,
                           Array<BoundaryCondition*>& bcs)
  : a(a), L(L), mesh(mesh), bcs(bcs)
{
  message("Creating nonlinear PDE with %d boundary condition(s).", bcs.size());

  // Check ranks of forms
  if ( a.form().rank() != 2 )
    error("Expected a bilinear form but rank is %d.", a.form().rank());
  if ( L.form().rank() != 1 )
    error("Expected a linear form but rank is %d.", L.form().rank());
}
//-----------------------------------------------------------------------------
NonlinearPDE::~NonlinearPDE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NonlinearPDE::update(const GenericVector& x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NonlinearPDE::form(GenericMatrix& A, GenericVector& b, const GenericVector& x)
{
  // Assemble 
  assembler.assemble(A, a, mesh);
  assembler.assemble(b, L, mesh);

  // Apply boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
    bcs[i]->apply(A, b, x, a);
}
//-----------------------------------------------------------------------------
void NonlinearPDE::solve(Function& u, real& t, const real& T, const real& dt)
{
  begin("Solving nonlinear PDE.");  

  // Initialise function
  u.init(mesh, x, a, 1);

  // Solve
  while( t < T )
  {
    t += dt;
    newton_solver.solve(*this ,x);
  }
}
//-----------------------------------------------------------------------------

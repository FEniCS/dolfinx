// Copyright (C) 2005-2008 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2006-2007.
//
// First added:  2005-10-24
// Last changed: 2008-09-03

#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Function.h>
#include <dolfin/fem/Form.h>
#include <dolfin/log/dolfin_log.h>
#include "NonlinearPDE.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NonlinearPDE::NonlinearPDE(Form& a, Form& L, Mesh& mesh, DirichletBC& bc)
  : a(a), L(L), mesh(mesh), assembler(mesh)
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
NonlinearPDE::NonlinearPDE(Form& a, Form& L, Mesh& mesh, 
  Array<DirichletBC*>& bcs) : a(a), L(L), mesh(mesh), bcs(bcs), assembler(mesh)
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
void NonlinearPDE::F(GenericVector& b, const GenericVector& x)
{
  // Assemble 
  assembler.assemble(b, L);

  // Apply boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
    bcs[i]->apply(b, x, a);
}
//-----------------------------------------------------------------------------
void NonlinearPDE::J(GenericMatrix& A, const GenericVector& x)
{
  // Assemble 
  assembler.assemble(A, a);

  // Apply boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
    bcs[i]->apply(A, a);
}
//-----------------------------------------------------------------------------
void NonlinearPDE::solve(Function& u, double& t, const double& T, const double& dt)
{
  begin("Solving nonlinear PDE.");  

  // Initialise function
  u.init(mesh, a, 1);
  GenericVector& x = u.vector();

  // Solve
  while( t < T )
  {
    t += dt;
    newton_solver.solve(*this ,x);
  }

  end();
}
//-----------------------------------------------------------------------------

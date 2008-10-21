// Copyright (C) 2004-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006-2008.
//
// First added:  2004
// Last changed: 2008-09-09

#include <tr1/memory>
#include <dolfin/fem/Assembler.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/Form.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/LUSolver.h>
#include <dolfin/la/KrylovSolver.h>
#include <dolfin/la/enums_la.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/io/dolfin_io.h>
#include "LinearPDE.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearPDE::LinearPDE(Form& a, Form& L, Mesh& mesh, MatrixType matrix_type)
                   : a(a), L(L), mesh(mesh), bcs(0), matrix_type(matrix_type)
{
  message("Creating linear PDE.");
}
//-----------------------------------------------------------------------------
LinearPDE::LinearPDE(Form& a, Form& L, Mesh& mesh, DirichletBC& bc, 
                     MatrixType matrix_type) : a(a), L(L), mesh(mesh), 
                     matrix_type(matrix_type)
{
  message("Creating linear PDE with one boundary condition.");
  bcs.push_back(&bc);
} 
//-----------------------------------------------------------------------------
LinearPDE::LinearPDE(Form& a, Form& L, Mesh& mesh, Array<DirichletBC*>& bcs, 
                     MatrixType matrix_type) : a(a), L(L), mesh(mesh), 
                     matrix_type(matrix_type)
{
  message("Creating linear PDE with %d boundary condition(s).", bcs.size());
  for (uint i = 0; i < bcs.size(); i++)
    this->bcs.push_back(bcs[i]);
}
//-----------------------------------------------------------------------------
LinearPDE::~LinearPDE()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void LinearPDE::solve(Function& u)
{
  begin("Solving linear PDE.");

  // Create matrix and vector for assembly
  Matrix A;
  Vector b;
  GenericVector& x = u.vector();

  // Assemble linear system and apply boundary conditions
  //Assembler assembler(mesh);  
  //assembler.assemble(A, a, b, L, bcs);

  // Assemble linear system and apply boundary conditions
  Assembler assembler(mesh);  
  assembler.assemble(A, a);
  assembler.assemble(b, L);
  for (uint i = 0; i < bcs.size(); i++)
    bcs[i]->apply(A, b, a);

  // Solve linear system
  const std::string solver_type = get("PDE linear solver");
  if ( solver_type == "direct" )
  {
    //LUSolver solver(matrix_type);
    LUSolver solver;
    solver.set("parent", *this);
    solver.solve(A, x, b);
  }
  else if ( solver_type == "iterative" )
  {
    KrylovSolver solver(gmres);
    solver.set("parent", *this);
    solver.solve(A, x, b);
/*
    if( matrix_type == symmetric)
    {
      KrylovSolver solver(cg);
      solver.set("parent", *this);
      solver.solve(A, *x, b);
    }
    else
    {
      KrylovSolver solver(gmres);
      solver.set("parent", *this);
      solver.solve(A, *x, b);
    }
*/
  }
  else
    error("Unknown solver type \"%s\".", solver_type.c_str());

  end();
}
//-----------------------------------------------------------------------------
void LinearPDE::solve(Function& u0, Function& u1)
{
  error("Need to fix LinearPDE for sub-functions.");
/*
  // Solve system
  Function u;
  solve(u);

  // Extract sub functions
  u0 = u[0];
  u1 = u[1];
*/
}
//-----------------------------------------------------------------------------
void LinearPDE::solve(Function& u0, Function& u1, Function& u2)
{
  error("Need to fix LinearPDE for sub-functions.");
/*
  // Solve system
  Function u;
  solve(u);

  // Extract sub functions
  u0 = u[0];
  u1 = u[1];
  u2 = u[2];
*/
}
//-----------------------------------------------------------------------------

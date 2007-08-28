// Copyright (C) 2004-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006
//
// First added:  2004
// Last changed: 2007-08-28

#include <dolfin/Matrix.h>
#include <dolfin/BoundaryCondition.h>
#include <dolfin/Assembler.h>
#include <dolfin/LUSolver.h>
#include <dolfin/KrylovSolver.h>
#include <dolfin/Function.h>
#include <dolfin/LinearPDE.h>
#include <dolfin/dolfin_io.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearPDE::LinearPDE(Form& a, Form& L, Mesh& mesh)
  : a(a), L(L), mesh(mesh)
{
  message("Creating linear PDE.");
}
//-----------------------------------------------------------------------------
LinearPDE::LinearPDE(Form& a, Form& L, Mesh& mesh, BoundaryCondition& bc)
  : a(a), L(L), mesh(mesh)
{
  message("Creating linear PDE with one boundary condition.");
  bcs.push_back(&bc);
} 
//-----------------------------------------------------------------------------
LinearPDE::LinearPDE(Form& a, Form& L, Mesh& mesh, Array<BoundaryCondition*>& bcs)
  : a(a), L(L), mesh(mesh)
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

  // Assemble linear system
  Assembler assembler(mesh);
  assembler.assemble(A, a);
  assembler.assemble(b, L);

  // Apply boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
    bcs[i]->apply(A, b, a);

  // Solve linear system
  const std::string solver_type = get("PDE linear solver");
  if ( solver_type == "direct" )
  {
    cout << "Using direct solver." << endl;
    LUSolver solver;
    solver.set("parent", *this);
    solver.solve(A, x, b);
  }
  else if ( solver_type == "iterative" )
  {
    cout << "Using iterative solver (GMRES)." << endl;
    KrylovSolver solver(gmres);
    solver.set("parent", *this);
    solver.solve(A, x, b);
  }
  else
    error("Unknown solver type \"%s\".", solver_type.c_str());

  //cout << "Matrix:" << endl;
  //A.disp();

  //cout << "Vector:" << endl;
  //b.disp();

  //cout << "Solution vector:" << endl;
  //x.disp();

  // Set function data
  u.init(mesh, x, a, 1);

  end();
}
//-----------------------------------------------------------------------------
void LinearPDE::solve(Function& u0, Function& u1)
{
  // Solve system
  Function u;
  solve(u);

  // Extract sub functions
  u0 = u[0];
  u1 = u[1];
}
//-----------------------------------------------------------------------------
void LinearPDE::solve(Function& u0, Function& u1, Function& u2)
{
  // Solve system
  Function u;
  solve(u);

  // Extract sub functions
  u0 = u[0];
  u1 = u[1];
  u2 = u[2];
}
//-----------------------------------------------------------------------------

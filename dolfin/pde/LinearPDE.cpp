// Copyright (C) 2004-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006-2008.
//
// First added:  2004
// Last changed: 2008-08-20

#include <dolfin/la/Matrix.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/Assembler.h>
#include <dolfin/la/LUSolver.h>
#include <dolfin/la/KrylovSolver.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/DiscreteFunction.h>
#include "LinearPDE.h"
#include <dolfin/io/dolfin_io.h>
#include <dolfin/fem/Form.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
LinearPDE::LinearPDE(Form& a, Form& L, Mesh& mesh)
  : a(a), L(L), mesh(mesh)
{
  message("Creating linear PDE.");
}
//-----------------------------------------------------------------------------
LinearPDE::LinearPDE(Form& a, Form& L, Mesh& mesh, DirichletBC& bc)
  : a(a), L(L), mesh(mesh)
{
  message("Creating linear PDE with one boundary condition.");
  bcs.push_back(&bc);
} 
//-----------------------------------------------------------------------------
LinearPDE::LinearPDE(Form& a, Form& L, Mesh& mesh, Array<DirichletBC*>& bcs)
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
  Vector* x = new Vector();

  // Assemble linear system and apply boundary conditions
  Assembler assembler(mesh);  
  assembler.assemble(A, a, b, L, bcs);

  // Assemble linear system and apply boundary conditions
  //Assembler assembler(mesh);  
  //assembler.assemble(A, a);
  //assembler.assemble(b, L);
  //for (uint i = 0; i < bcs.size(); i++)
  //  bcs[i]->apply(A, b, a);

  // Solve linear system
  const std::string solver_type = get("PDE linear solver");
  if ( solver_type == "direct" )
  {
    LUSolver solver;
    solver.set("parent", *this);
    solver.solve(A, *x, b);
  }
  else if ( solver_type == "iterative" )
  {
    KrylovSolver solver(gmres);
    solver.set("parent", *this);
    solver.solve(A, *x, b);
  }
  else
    error("Unknown solver type \"%s\".", solver_type.c_str());

  //cout << "Matrix:" << endl;
  //A.disp();

  //cout << "Vector:" << endl;
  //b.disp();

  //cout << "Solution vector:" << endl;
  //x.disp();

  u.init(mesh, *x, a, 1);
  DiscreteFunction& uu = dynamic_cast<DiscreteFunction&>(*u.f);
  uu.local_vector = x;

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

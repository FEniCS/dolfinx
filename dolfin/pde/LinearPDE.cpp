// Copyright (C) 2004-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006-2008.
//
// First added:  2004
// Last changed: 2008-11-15

#include <tr1/memory>
#include <dolfin/fem/Assembler.h>
#include <dolfin/fem/BoundaryCondition.h>
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
LinearPDE::LinearPDE(Form& a, Form& L, MatrixType matrix_type)
                   : a(a), L(L), bcs(0), matrix_type(matrix_type)
{
  message("Creating linear PDE.");
}
//-----------------------------------------------------------------------------
LinearPDE::LinearPDE(Form& a, Form& L, BoundaryCondition& bc, 
                     MatrixType matrix_type) : a(a), L(L), 
                     matrix_type(matrix_type)
{
  message("Creating linear PDE with one boundary condition.");
  bcs.push_back(&bc);
} 
//-----------------------------------------------------------------------------
LinearPDE::LinearPDE(Form& a, Form& L, Array<BoundaryCondition*>& bcs, 
                     MatrixType matrix_type) : a(a), L(L), 
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

  // Set function space if missing
  if (!u.has_function_space())
  {
    dolfin_assert(a._function_spaces.size() == 2);
    u._function_space = a._function_spaces[1];
  }

  // Create matrix and vector for assembly
  Matrix A;
  Vector b;
  GenericVector& x = u.vector();

  // Assemble linear system and apply boundary conditions
  if( matrix_type == symmetric)
  {
    cout << "Symm assembly " << endl;
    std::vector<const DirichletBC*> _bcs;
    for(uint i=0; i< bcs.size(); ++i)
    {
      const DirichletBC* _bc = dynamic_cast<const DirichletBC*>(bcs[i]);
      if (!_bc)
        error("Error casting to DirichletBC in LinearPDE");    
      _bcs.push_back(_bc);
    }
    Assembler::assemble(A, a, b, L, _bcs);
  }
  else
  {
    Assembler::assemble(A, a);
    Assembler::assemble(b, L);
    for (uint i = 0; i < bcs.size(); i++)
      bcs[i]->apply(A, b);
  }

  // Solve linear system
  const std::string solver_type = get("PDE linear solver");
  if (solver_type == "direct")
  {
    LUSolver solver(matrix_type);
    solver.set("parent", *this);
    solver.solve(A, x, b);
  }
  else if (solver_type == "iterative")
  {
    if( matrix_type == symmetric)
    {
      KrylovSolver solver(cg);
      solver.set("parent", *this);
      solver.solve(A, x, b);
    }
    else
    {
      KrylovSolver solver(gmres);
      solver.set("parent", *this);
      solver.solve(A, x, b);
    }
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

// Copyright (C) 2008 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-12-26
// Last changed: 2008-12-26

#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/LUSolver.h>
#include <dolfin/la/KrylovSolver.h>
#include <dolfin/nls/NewtonSolver.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/SubFunction.h>
#include "assemble.h"
#include "BoundaryCondition.h"
#include "VariationalProblem.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& a,
                                       const Form& L,
                                       bool nonlinear)
  : a(a), L(L), nonlinear(nonlinear), newton_solver(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& a,
                                       const Form& L,
                                       const BoundaryCondition& bc,
                                       bool nonlinear)
  : a(a), L(L), nonlinear(nonlinear), newton_solver(0)
{
  bcs.push_back(&bc);
}
//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& a,
                                       const Form& L,
                                       std::vector<BoundaryCondition*>& bcs,
                                       bool nonlinear)
  : a(a), L(L), nonlinear(nonlinear), newton_solver(0)
{
  for (uint i = 0; i < bcs.size(); i++)
    this->bcs.push_back(bcs[i]);
}
//-----------------------------------------------------------------------------
VariationalProblem::~VariationalProblem()
{
  delete newton_solver;
}
//-----------------------------------------------------------------------------
void VariationalProblem::solve(Function& u)
{
  // Solve linear or nonlinear variational problem
  if (nonlinear)
    solve_nonlinear(u);
  else
    solve_linear(u);
}
//-----------------------------------------------------------------------------
void VariationalProblem::solve(Function& u0, Function& u1)
{
  // Solve variational problem
  Function u;
  solve(u);

  // Extract subfunctions
  u0 = u[0];
  u1 = u[1];
}
//-----------------------------------------------------------------------------
void VariationalProblem::solve(Function& u0, Function& u1, Function& u2)
{
  // Solve variational problem
  Function u;
  solve(u);

  // Extract subfunctions
  u0 = u[0];
  u1 = u[1];
  u2 = u[2];
}
//-----------------------------------------------------------------------------
void VariationalProblem::F(GenericVector& b, const GenericVector& x)
{
  // Assemble
  assemble(b, L);

  // Apply boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
    bcs[i]->apply(b, x);
}
//-----------------------------------------------------------------------------
void VariationalProblem::J(GenericMatrix& A, const GenericVector& x)
{
  // Assemble
  assemble(A, a);

  // Apply boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
    bcs[i]->apply(A);
}
//-----------------------------------------------------------------------------
void VariationalProblem::update(const GenericVector& x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void VariationalProblem::solve_linear(Function& u)
{
  begin("Solving linear variational problem");

  // Set function space if missing
  if (!u.has_function_space())
  {
    dolfin_assert(a._function_spaces.size() == 2);
    u._function_space = a._function_spaces[1];
  }

  // Assemble linear system
  Matrix A;
  Vector b;
  assemble(A, a);
  assemble(b, L);

  // Apply boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
    bcs[i]->apply(A, b);

  // Solve linear system
  const std::string solver_type = get("linear solver");
  if (solver_type == "direct")
  {
    LUSolver solver;
    solver.set("parent", *this);
    solver.solve(A, u.vector(), b);
  }
  else if (solver_type == "iterative")
  {
    KrylovSolver solver(gmres);
    solver.set("parent", *this);
    solver.solve(A, u.vector(), b);
  }
  else
    error("Unknown solver type \"%s\".", solver_type.c_str());

  end();
}
//-----------------------------------------------------------------------------
void VariationalProblem::solve_nonlinear(Function& u)
{
  begin("Solving nonlinear variational problem");

  // Set function space if missing
  if (!u.has_function_space())
  {
    dolfin_assert(a._function_spaces.size() == 2);
    u._function_space = a._function_spaces[1];
  }

  // Create Newton solver if missing
  if (!newton_solver)
    newton_solver = new NewtonSolver();

  // Call Newton solver
  dolfin_assert(newton_solver);
  newton_solver->solve(*this, u.vector());

  end();
}
//-----------------------------------------------------------------------------

// Copyright (C) 2008-2009 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-12-26
// Last changed: 2009-03-06

#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/LUSolver.h>
#include <dolfin/la/KrylovSolver.h>
#include <dolfin/nls/NewtonSolver.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/SubFunction.h>
#include "assemble.h"
#include "Form.h"
#include "BoundaryCondition.h"
#include "DirichletBC.h"
#include "VariationalProblem.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& a,
                                       const Form& L,
                                       bool nonlinear)
  : a(a), L(L), cell_domains(0), exterior_facet_domains(0),
    interior_facet_domains(0), nonlinear(nonlinear), _newton_solver(0)

{
  // FIXME: Must be set in DefaultParameters.h because of bug in cross-platform parameter system
  // Add parameter "symmetric"
  //add("symmetric", false);
}
//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& a,
                                       const Form& L,
                                       const BoundaryCondition& bc,
                                       bool nonlinear)
  : a(a), L(L), cell_domains(0), exterior_facet_domains(0),
    interior_facet_domains(0), nonlinear(nonlinear), _newton_solver(0)
{
  // Store boundary condition
  bcs.push_back(&bc);

  // FIXME: Must be set in DefaultParameters.h because of bug in cross-platform parameter system
  // Add parameter "symmetric"
  //add("symmetric", false);
}
//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& a,
                                       const Form& L,
                                       std::vector<BoundaryCondition*>& bcs,
                                       bool nonlinear)
  : a(a), L(L), cell_domains(0), exterior_facet_domains(0),
    interior_facet_domains(0), nonlinear(nonlinear), _newton_solver(0)
{
  // Store boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
    this->bcs.push_back(bcs[i]);

  // FIXME: Must be set in DefaultParameters.h because of bug in cross-platform parameter system
  // Add parameter "symmetric"
  //add("symmetric", false);
}
//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& a,
                                       const Form& L,
                                       std::vector<BoundaryCondition*>& bcs,
                                       const MeshFunction<uint>* cell_domains,
                                       const MeshFunction<uint>* exterior_facet_domains,
                                       const MeshFunction<uint>* interior_facet_domains,
                                       bool nonlinear)
  : a(a), L(L), cell_domains(cell_domains),
    exterior_facet_domains(exterior_facet_domains),
    interior_facet_domains(interior_facet_domains), nonlinear(nonlinear),
    _newton_solver(0)
{
  // Store boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
    this->bcs.push_back(bcs[i]);

  // FIXME: Must be set in DefaultParameters.h because of bug in cross-platform parameter system
  // Add parameter "symmetric"
  //add("symmetric", false);
}
//-----------------------------------------------------------------------------
VariationalProblem::~VariationalProblem()
{
  delete _newton_solver;
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
  // Check that we are solving a nonlinear problem
  if (!nonlinear)
    error("Attempt to solve linear variational problem with Newton solver.");

  // Assemble
  assemble(b, L, cell_domains, exterior_facet_domains, interior_facet_domains);

  // Apply boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
    bcs[i]->apply(b, x);
}
//-----------------------------------------------------------------------------
void VariationalProblem::J(GenericMatrix& A, const GenericVector& x)
{
  // Check that we are solving a nonlinear problem
  if (!nonlinear)
    error("Attempt to solve linear variational problem with Newton solver.");

  // Assemble
  assemble(A, a, cell_domains, exterior_facet_domains, interior_facet_domains);

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
NewtonSolver& VariationalProblem::newton_solver()
{
  // Create Newton solver if missing
  if (!_newton_solver)
  {
    _newton_solver = new NewtonSolver();
    _newton_solver->set("parent", *this);
  }

  dolfin_assert(_newton_solver);
  return *_newton_solver;
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

  // Check if system is symmetric
  const bool symmetric = get("symmetric");

  // Create matrix and vector
  Matrix A;
  Vector b;

  // Different assembly depending on whether or not the system is symmetric
  if (symmetric)
  {
    // Need to cast to DirichletBC to use assemble_system
    std::vector<const DirichletBC*> _bcs;
    for (uint i = 0; i < bcs.size(); i++)
    {
      const DirichletBC* _bc = dynamic_cast<const DirichletBC*>(bcs[i]);
      if (!_bc)
        error("Only Dirichlet boundary conditions may be used for assembly of symmetric system.");
      _bcs.push_back(_bc);
    }

    // Assemble linear system and apply boundary conditions
    assemble_system(A, b, a, L, _bcs, cell_domains, exterior_facet_domains, interior_facet_domains, 0);
  }
  else
  {
    // Assemble linear system
    assemble(A, a, cell_domains, exterior_facet_domains, interior_facet_domains);
    summary();
    assemble(b, L, cell_domains, exterior_facet_domains, interior_facet_domains);

    // Apply boundary conditions
    for (uint i = 0; i < bcs.size(); i++)
      bcs[i]->apply(A, b);
  }

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
    if ("symmetric")
    {
      KrylovSolver solver("gmres");
      solver.set("parent", *this);
      solver.solve(A, u.vector(), b);
    }
    else
    {
      KrylovSolver solver("cg");
      solver.set("parent", *this);
      solver.solve(A, u.vector(), b);
    }
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

  // Call Newton solver
  newton_solver().solve(*this, u.vector());

  end();
}
//-----------------------------------------------------------------------------

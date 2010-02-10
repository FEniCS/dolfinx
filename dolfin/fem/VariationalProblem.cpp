// Copyright (C) 2008-2009 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-12-26
// Last changed: 2009-10-06

#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/LUSolver.h>
#include <dolfin/la/KrylovSolver.h>
#include <dolfin/nls/NewtonSolver.h>
#include <dolfin/function/Function.h>
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
    interior_facet_domains(0), nonlinear(nonlinear), jacobian_initialised(false), 
    _newton_solver(0)

{
  // Set default parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& a,
                                       const Form& L,
                                       const BoundaryCondition& bc,
                                       bool nonlinear)
  : a(a), L(L), cell_domains(0), exterior_facet_domains(0),
    interior_facet_domains(0), nonlinear(nonlinear), jacobian_initialised(false), 
    _newton_solver(0)
{
  // Set default parameters
  parameters = default_parameters();

  // Store boundary condition
  bcs.push_back(&bc);
}
//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& a,
                                       const Form& L,
                                       const std::vector<const BoundaryCondition*>& bcs,
                                       bool nonlinear)
  : a(a), L(L), cell_domains(0), exterior_facet_domains(0),
    interior_facet_domains(0), nonlinear(nonlinear), jacobian_initialised(false),
    _newton_solver(0)
{
  // Set default parameters
  parameters = default_parameters();

  // Store boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
    this->bcs.push_back(bcs[i]);
}
//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& a,
                                       const Form& L,
                                       const std::vector<const BoundaryCondition*>& bcs,
                                       const MeshFunction<uint>* cell_domains,
                                       const MeshFunction<uint>* exterior_facet_domains,
                                       const MeshFunction<uint>* interior_facet_domains,
                                       bool nonlinear)
  : a(a), L(L), cell_domains(cell_domains),
    exterior_facet_domains(exterior_facet_domains),
    interior_facet_domains(interior_facet_domains), nonlinear(nonlinear), 
    jacobian_initialised(false), _newton_solver(0)
{
  // Set default parameters
  parameters = default_parameters();

  // Store boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
    this->bcs.push_back(bcs[i]);
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
  // Create function 
  Function u(a.function_space(0));
  
  // Solve variational problem
  solve(u);

  // Extract subfunctions
  u0 = u[0];
  u1 = u[1];
}
//-----------------------------------------------------------------------------
void VariationalProblem::solve(Function& u0, Function& u1, Function& u2)
{
  // Create function 
  Function u(a.function_space(0));

  // Solve variational problem
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

  // Check if Jacobian matrix sparsity pattern should be reset 
  bool reset_sparsity = true;
  if (parameters["reset_jacobian"] && jacobian_initialised) 
    reset_sparsity = false;

  // Assemble
  assemble(A, a, cell_domains, exterior_facet_domains, interior_facet_domains, 
           reset_sparsity);
  jacobian_initialised = true;

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
    _newton_solver->parameters.update(parameters("newton_solver"));
  }

  assert(_newton_solver);
  return *_newton_solver;
}
//-----------------------------------------------------------------------------
void VariationalProblem::solve_linear(Function& u)
{
  begin("Solving linear variational problem");

  // Check if system is symmetric
  const bool symmetric = parameters["symmetric"];

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
    assemble_system(A, b, a, L, _bcs, cell_domains, exterior_facet_domains, interior_facet_domains, 0, true);
  }
  else
  {
    // Assemble linear system
    assemble(A, a, cell_domains, exterior_facet_domains, interior_facet_domains);
    assemble(b, L, cell_domains, exterior_facet_domains, interior_facet_domains);

    // Apply boundary conditions
    for (uint i = 0; i < bcs.size(); i++)
      bcs[i]->apply(A, b);
  }

  // Solve linear system
  const std::string solver_type = parameters["linear_solver"];
  if (solver_type == "direct")
  {
    LUSolver solver;
    solver.parameters.update(parameters("lu_solver"));
    solver.solve(A, u.vector(), b);
  }
  else if (solver_type == "iterative")
  {
    if (symmetric)
    {
      KrylovSolver solver("cg");
      solver.parameters.update(parameters("krylov_solver"));
      solver.solve(A, u.vector(), b);
    }
    else
    {
      KrylovSolver solver("gmres");
      solver.parameters.update(parameters("krylov_solver"));
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

  // Call Newton solver
  newton_solver().solve(*this, u.vector());

  end();
}
//-----------------------------------------------------------------------------

// Copyright (C) 2008-2009 Anders Logg and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Marie E. Rognes 2011
//
// First added:  2008-12-26
// Last changed: 2011-01-14

#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/LUSolver.h>
#include <dolfin/la/KrylovSolver.h>
#include <dolfin/nls/NewtonSolver.h>
#include <dolfin/function/Function.h>
#include <dolfin/adaptivity/AdaptiveSolver.h>
#include <dolfin/adaptivity/GoalFunctional.h>
#include "assemble.h"
#include "Form.h"
#include "BoundaryCondition.h"
#include "DirichletBC.h"
#include "VariationalProblem.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& a, const Form& L)
  : a(extract_bilinear(a, L)), L(extract_linear(a, L)),
    _cell_domains(0), _exterior_facet_domains(0), _interior_facet_domains(0),
    nonlinear(is_nonlinear(a, L)), jacobian_initialised(false)
{
  // Set default parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& a, const Form& L,
                                       const BoundaryCondition& bc)
  : a(extract_bilinear(a, L)), L(extract_linear(a, L)),
    _cell_domains(0), _exterior_facet_domains(0), _interior_facet_domains(0),
    nonlinear(is_nonlinear(a, L)), jacobian_initialised(false)
{
  // Set default parameters
  parameters = default_parameters();

  // Store boundary condition
  _bcs.push_back(&bc);
}
//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& a,
                                       const Form& L,
                                       const std::vector<const BoundaryCondition*>& bcs)
  : a(extract_bilinear(a, L)), L(extract_linear(a, L)),
    _cell_domains(0), _exterior_facet_domains(0), _interior_facet_domains(0),
    nonlinear(is_nonlinear(a, L)), jacobian_initialised(false)
{
  // Set default parameters
  parameters = default_parameters();

  // Store boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
    this->_bcs.push_back(bcs[i]);
}
//-----------------------------------------------------------------------------
VariationalProblem::VariationalProblem(const Form& a,
                                       const Form& L,
                                       const std::vector<const BoundaryCondition*>& bcs,
                                       const MeshFunction<uint>* cell_domains,
                                       const MeshFunction<uint>* exterior_facet_domains,
                                       const MeshFunction<uint>* interior_facet_domains)
  : a(extract_bilinear(a, L)), L(extract_linear(a, L)),
    _cell_domains(cell_domains),
    _exterior_facet_domains(exterior_facet_domains),
    _interior_facet_domains(interior_facet_domains),
    nonlinear(is_nonlinear(a, L)), jacobian_initialised(false)
{
  // Set default parameters
  parameters = default_parameters();

  // Store boundary conditions
  for (uint i = 0; i < bcs.size(); i++)
    this->_bcs.push_back(bcs[i]);
}
//-----------------------------------------------------------------------------
VariationalProblem::~VariationalProblem()
{
  // Do nothing
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
void VariationalProblem::solve(Function& u, const double tol, GoalFunctional& M)
{
  // Call adaptive solver
  AdaptiveSolver::solve(u, *this, tol, M, parameters("adaptive_solver"));
}
//-----------------------------------------------------------------------------
void VariationalProblem::solve(Function& u, const double tol, Form& M,
                               ErrorControl& ec)
{
  // Call adaptive solver
  AdaptiveSolver::solve(u, *this, tol, M, ec, parameters("adaptive_solver"));
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
  assemble(b, L, _cell_domains, _exterior_facet_domains, _interior_facet_domains);

  // Apply boundary conditions
  for (uint i = 0; i < _bcs.size(); i++)
    _bcs[i]->apply(b, x);

  // Print vector
  const bool print_rhs = parameters["print_rhs"];
  if (print_rhs == true)
    info(b, true);
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
  assemble(A, a, _cell_domains, _exterior_facet_domains, _interior_facet_domains,
           reset_sparsity);
  jacobian_initialised = true;

  // Apply boundary conditions
  for (uint i = 0; i < _bcs.size(); i++)
    _bcs[i]->apply(A);

  // Print matrix
  const bool print_matrix = parameters["print_matrix"];
  if (print_matrix == true)
    info(A, true);
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
    _newton_solver.reset(new NewtonSolver);
    _newton_solver->parameters.update(parameters("newton_solver"));
  }

  assert(_newton_solver);
  return *_newton_solver;
}
//-----------------------------------------------------------------------------
void VariationalProblem::solve_linear(Function& u)
{
  begin("Solving linear variational problem");

  // Get parameters
  std::string solver_type   = parameters["linear_solver"];
  const std::string pc_type = parameters["preconditioner"];
  const bool symmetric      = parameters["symmetric"];
  const bool print_rhs      = parameters["print_rhs"];
  const bool print_matrix   = parameters["print_matrix"];

  // Create matrix and vector
  Matrix A;
  Vector b;

  // Different assembly depending on whether or not the system is symmetric
  if (symmetric)
  {
    // Need to cast to DirichletBC to use assemble_system
    std::vector<const DirichletBC*> __bcs;
    for (uint i = 0; i < _bcs.size(); i++)
    {
      const DirichletBC* _bc = dynamic_cast<const DirichletBC*>(_bcs[i]);
      if (!_bc)
        error("Only Dirichlet boundary conditions may be used for assembly of symmetric system.");
      __bcs.push_back(_bc);
    }

    // Assemble linear system and apply boundary conditions
    assemble_system(A, b, a, L, __bcs, _cell_domains, _exterior_facet_domains, _interior_facet_domains, 0, true);
  }
  else
  {
    // Assemble linear system
    assemble(A, a, _cell_domains, _exterior_facet_domains, _interior_facet_domains);
    assemble(b, L, _cell_domains, _exterior_facet_domains, _interior_facet_domains);

    // Apply boundary conditions
    for (uint i = 0; i < _bcs.size(); i++)
      _bcs[i]->apply(A, b);
  }

  // Print vector/matrix
  if (print_rhs == true)
    info(b, true);
  if (print_matrix == true)
    info(A, true);

  // Adjust solver type if necessary
  if (solver_type == "iterative")
  {
    if (symmetric)
      solver_type = "cg";
    else
      solver_type = "gmres";
  }

  //  Solve linear system
  if (solver_type == "lu" || solver_type == "direct")
  {
    LUSolver solver;
    solver.parameters.update(parameters("lu_solver"));
    solver.solve(A, u.vector(), b);
  }
  else
  {
    KrylovSolver solver(solver_type, pc_type);
    solver.parameters.update(parameters("krylov_solver"));
    solver.solve(A, u.vector(), b);
  }

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
//-----------------------------------------------------------------------------
const Form& VariationalProblem::extract_linear(const Form& b, const Form& c) const
{
  // Return the argument that has rank 1 (and check that other
  // argument has rank 2)
  if (c.rank() == 1 && b.rank() == 2)
    return c;

  if (b.rank() == 1 && c.rank() == 2)
    return b;

  error("This is not a valid combination of a bilinear and a linear form");
  return b;
}
//-----------------------------------------------------------------------------
const Form& VariationalProblem::extract_bilinear(const Form& b, const Form& c) const
{
  // Return the argument that has rank 2 (and check that other
  // argument has rank 1)
  if (c.rank() == 2 && b.rank() == 1)
    return c;

  if (b.rank() == 2 && c.rank() == 1)
    return b;

  error("This is not a valid combination of a bilinear and a linear form");
  return c;
}
//-----------------------------------------------------------------------------
bool VariationalProblem::is_nonlinear(const Form& b, const Form& c) const
{
  // Linear if the first argument is the linear form.
  if (b.rank() == 1)
    return true;
  return false;
}
//-----------------------------------------------------------------------------
const bool VariationalProblem::is_nonlinear() const
{
	return nonlinear;
}
//-----------------------------------------------------------------------------

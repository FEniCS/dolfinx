// Copyright (C) 2005-2016 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Anders Logg 2011
//
// This program illustrates the use of the DOLFIN nonlinear solver for solving
// the Cahn-Hilliard equation.
//
// The Cahn-Hilliard equation is very sensitive to the chosen parameters and
// time step. It also requires fines meshes, and is often not well-suited to
// iterative linear solvers.

// Begin demo

#include <dolfin.h>
#include "CahnHilliard2D.h"
#include "CahnHilliard3D.h"

using namespace dolfin;

// Initial conditions
class InitialConditions : public Expression
{
public:

  InitialConditions() : Expression(2)
  { dolfin::seed(2 + dolfin::MPI::rank(MPI_COMM_WORLD)); }

  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0]= 0.63 + 0.02*(0.5 - dolfin::rand());
    values[1]= 0.0;
  }

};

// User defined nonlinear problem
class CahnHilliardEquation : public NonlinearProblem
{
  public:

    // Constructor
    CahnHilliardEquation(std::shared_ptr<const Form> F,
                         std::shared_ptr<const Form> J) : _F(F), _J(J) {}

    // User defined residual vector
    void F(GenericVector& b, const GenericVector& x)
    { assemble(b, *_F); }

    // User defined assemble of Jacobian
    void J(GenericMatrix& A, const GenericVector& x)
    { assemble(A, *_J); }

  private:

    // Forms
    std::shared_ptr<const Form> _F;
    std::shared_ptr<const Form> _J;
};


int main(int argc, char* argv[])
{
  init(argc, argv);

  // Mesh
  auto mesh = std::make_shared<UnitSquareMesh>(96, 96);

  // Create function space and forms, depending on spatial dimension
  // of the mesh
  std::shared_ptr<FunctionSpace> V;
  std::shared_ptr<Form> F, J;
  if (mesh->geometry().dim() == 2)
  {
    V = std::make_shared<CahnHilliard2D::FunctionSpace>(mesh);
    F = std::make_shared<CahnHilliard2D::ResidualForm>(V);
    J = std::make_shared<CahnHilliard2D::JacobianForm>(V, V);
  }
  else if(mesh->geometry().dim() == 3)
  {
    V = std::make_shared<CahnHilliard3D::FunctionSpace>(mesh);
    F = std::make_shared<CahnHilliard3D::ResidualForm>(V);
    J = std::make_shared<CahnHilliard3D::JacobianForm>(V, V);
  }
  else
    error("This demo only supports two or three spatial dimensions.");

  // Create solution Functions (at t_n and t_{n+1})
  auto u0 = std::make_shared<Function>(V);
  auto u = std::make_shared<Function>(V);

  // Set solution to intitial condition
  InitialConditions u_initial;
  *u0 = u_initial;
  *u = u_initial;

  // Time stepping and model parameters
  auto dt = std::make_shared<Constant>(5.0e-6);
  auto theta = std::make_shared<Constant>(0.5);
  auto lambda = std::make_shared<Constant>(1.0e-2);

  // Collect coefficient into groups
  std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients
    = {{"u", u}, {"lmbda", lambda}, {"dt", dt}, {"theta", theta}};

  // Add extra coefficient for residual
  std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients_F = coefficients;
  coefficients_F.insert({"u0", u0});

  // Attach coefficients to form
  J->set_coefficients(coefficients);
  F->set_coefficients(coefficients_F);

  double t = 0.0;
  double T = 50*(*dt);

  // Create user-defined nonlinear problem
  CahnHilliardEquation cahn_hilliard(F, J);

  // Create nonlinear solver and set parameters
  NewtonSolver newton_solver;
  newton_solver.parameters["linear_solver"] = "lu";
  newton_solver.parameters["convergence_criterion"] = "incremental";
  newton_solver.parameters["maximum_iterations"] = 10;
  newton_solver.parameters["relative_tolerance"] = 1e-6;
  newton_solver.parameters["absolute_tolerance"] = 1e-15;

  // Save initial condition to file
  File file("cahn_hilliard.pvd", "compressed");
  file << (*u)[0];

  // Solve
  while (t < T)
  {
    // Update for next time step
    t += *dt;
    *u0->vector() = *u->vector();

    // Solve
    newton_solver.solve(cahn_hilliard, *u->vector());

    // Save function to file
    file << std::pair<const Function*, double>(&((*u)[0]), t);
  }

  // Plot solution
  plot((*u)[0]);
  interactive();

  return 0;
}

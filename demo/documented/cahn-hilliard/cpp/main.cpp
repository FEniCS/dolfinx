// Copyright (C) 2005-2007 Garth N. Wells
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
// First added:  2006-03-02
// Last changed: 2013-11-20
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
  {
    dolfin::seed(2 + dolfin::MPI::rank(MPI_COMM_WORLD));
  }

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
  CahnHilliardEquation(std::shared_ptr<const Mesh> mesh,
                       std::shared_ptr<const Constant> dt,
                       std::shared_ptr<const Constant> theta,
                       std::shared_ptr<const Constant> lambda)
    {
      // Initialize class (depending on geometric dimension of the mesh).
      // Unfortunately C++ does not allow namespaces as template arguments
      dolfin_assert(mesh);
      if (mesh->geometry().dim() == 2)
      {
        init<CahnHilliard2D::FunctionSpace, CahnHilliard2D::JacobianForm,
             CahnHilliard2D::ResidualForm>(mesh, dt, theta, lambda);
      }
      else if (mesh->geometry().dim() == 3)
      {
        init<CahnHilliard3D::FunctionSpace, CahnHilliard3D::JacobianForm,
             CahnHilliard3D::ResidualForm>(mesh, dt, theta, lambda);
      }
      else
        error("Cahn-Hilliard model is programmed for 2D and 3D only.");
    }

    // User defined residual vector
    void F(GenericVector& b, const GenericVector& x)
    {
      // Assemble RHS (Neumann boundary conditions)
      Assembler assembler;
      assembler.assemble(b, *L);
    }

    // User defined assemble of Jacobian
    void J(GenericMatrix& A, const GenericVector& x)
    {
      // Assemble system
      Assembler assembler;
      assembler.assemble(A, *a);
    }

    // Return solution function
    Function& u()
    { return *_u; }

    // Return solution function
    Function& u0()
    { return *_u0; }

  private:

    template<class X, class Y, class Z>
    void init(std::shared_ptr<const Mesh> mesh,
              std::shared_ptr<const Constant> dt,
              std::shared_ptr<const Constant> theta,
              std::shared_ptr<const Constant> lambda)
    {
      // Create function space and functions
      std::shared_ptr<X> V(new X(mesh));
      _u.reset(new Function(V));
      _u0.reset(new Function(V));

      // Create forms and attach functions
      Y* _a = new Y(V, V);
      Z* _L = new Z(V);
      _a->u = _u;
      _a->lmbda = lambda; _a->dt = dt; _a->theta = theta;
      _L->u = _u; _L->u0 = _u0;
      _L->lmbda = lambda; _L->dt = dt; _L->theta = theta;

      // Wrap pointers in a smart pointer
      a.reset(_a);
      L.reset(_L);

      // Set solution to intitial condition
      InitialConditions u_initial;
      *_u = u_initial;
    }

    // Function space, forms and functions
    std::unique_ptr<Form> a;
    std::unique_ptr<Form> L;
    std::shared_ptr<Function> _u;
    std::shared_ptr<Function> _u0;
};


int main(int argc, char* argv[])
{
  init(argc, argv);

  // Mesh
  auto mesh = std::make_shared<UnitSquareMesh>(96, 96);

  // Time stepping and model parameters
  auto dt = std::make_shared<Constant>(5.0e-6);
  auto theta = std::make_shared<Constant>(0.5);
  auto lambda = std::make_shared<Constant>(1.0e-2);

  double t = 0.0;
  double T = 50*(*dt);

  // Create user-defined nonlinear problem
  CahnHilliardEquation cahn_hilliard(mesh, dt, theta, lambda);

  // Solution functions
  Function& u = cahn_hilliard.u();
  Function& u0 = cahn_hilliard.u0();

  // Create nonlinear solver and set parameters
  NewtonSolver newton_solver;
  newton_solver.parameters["linear_solver"] = "lu";
  newton_solver.parameters["convergence_criterion"] = "incremental";
  newton_solver.parameters["maximum_iterations"] = 10;
  newton_solver.parameters["relative_tolerance"] = 1e-6;
  newton_solver.parameters["absolute_tolerance"] = 1e-15;

  // Save initial condition to file
  File file("cahn_hilliard.pvd", "compressed");
  file << u[0];

  // Solve
  while (t < T)
  {
    // Update for next time step
    t += *dt;
    *u0.vector() = *u.vector();

    // Solve
    newton_solver.solve(cahn_hilliard, *u.vector());

    // Save function to file
    file << std::pair<const Function*, double>(&(u[0]), t);
  }

  // Plot solution
  plot(u[0]);
  interactive();

  return 0;
}

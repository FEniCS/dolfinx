// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-03-02
// Last changed: 2010-09-01
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

  InitialConditions(const Mesh& mesh) : Expression(mesh.topology().dim())
  {
    dolfin::seed(2 + dolfin::MPI::process_number());
  }

  void eval(Array<double>& values, const Data& data) const
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
    CahnHilliardEquation(const Mesh& mesh, const Constant& dt,
                         const Constant& theta, const Constant& lambda)
                       : reset_Jacobian(true)
    {
      if (mesh.geometry().dim() == 2)
      {
        // Create function space and functions
        boost::shared_ptr<CahnHilliard2D::FunctionSpace> V(new CahnHilliard2D::FunctionSpace(mesh));
        _u.reset(new Function(V));
        _u0.reset(new Function(V));

        // Create forms and attach functions
        CahnHilliard2D::BilinearForm* _a = new CahnHilliard2D::BilinearForm(V, V);
        CahnHilliard2D::LinearForm*_L = new CahnHilliard2D::LinearForm(V);
        _a->u = *_u; _a->lmbda = lambda; _a->dt = dt; _a->theta = theta;
        _L->u = *_u; _L->u0 = *_u0;
        _L->lmbda = lambda; _L->dt = dt; _L->theta = theta;

        // Wrap pointers in a smart pointer
        a.reset(_a);
        L.reset(_L);

        // Set solution to intitial condition
        InitialConditions u_initial(mesh);
        *_u = u_initial;
      }
      else if (mesh.geometry().dim() == 3)
      {
        // Create function space and functions
        boost::shared_ptr<CahnHilliard3D::FunctionSpace> V(new CahnHilliard3D::FunctionSpace(mesh));
        _u.reset(new Function(V));
        _u0.reset(new Function(V));

        // Create forms and attach functions
        CahnHilliard3D::BilinearForm* _a = new CahnHilliard3D::BilinearForm(V, V);
        CahnHilliard3D::LinearForm*_L = new CahnHilliard3D::LinearForm(V);
        _a->u = *_u; _a->lmbda = lambda; _a->dt = dt; _a->theta = theta;
        _L->u = *_u; _L->u0 = *_u0;
        _L->lmbda = lambda; _L->dt = dt; _L->theta = theta;

        // Wrap pointers in a smart pointer
        a.reset(_a);
        L.reset(_L);

        // Set solution to intitial condition
        InitialConditions u_initial(mesh);
        *_u = u_initial;
      }
      else
        error("Cahn-Hilliard model is programmed for 2D and 3D only");
    }

    // User defined residual vector
    void F(GenericVector& b, const GenericVector& x)
    {
      // Assemble RHS (Neumann boundary conditions)
      assemble(b, *L);
    }

    // User defined assemble of Jacobian
    void J(GenericMatrix& A, const GenericVector& x)
    {
      // Assemble system and RHS (Neumann boundary conditions)
      assemble(A, *a, reset_Jacobian);
      reset_Jacobian  = false;
    }

    // Return solution function
    Function& u()
    { return *_u; }

    // Return solution function
    Function& u0()
    { return *_u0; }

  private:

    // Pointers to FunctionSpace and forms
    boost::scoped_ptr<Form> a;
    boost::scoped_ptr<Form> L;
    boost::scoped_ptr<Function> _u;
    boost::scoped_ptr<Function> _u0;
    bool reset_Jacobian;
};

int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);

  // Mesh
  UnitSquare mesh(96, 96);

  // Time stepping and model parameters
  Constant dt(5.0e-6);
  Constant theta(0.5);
  Constant lambda(1.0e-2);

  double t = 0.0;
  double T = 50*dt;

  // Create user-defined nonlinear problem
  CahnHilliardEquation cahn_hilliard(mesh, dt, theta, lambda);

  // Solution functions
  Function& u = cahn_hilliard.u();
  Function& u0 = cahn_hilliard.u0();

  // Create nonlinear solver and set parameters
  NewtonSolver newton_solver("lu");
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
    t += dt;
    u0.vector() = u.vector();

    // Solve
    newton_solver.solve(cahn_hilliard, u.vector());

    // Save function to file
    file << std::make_pair(&(u[0]), t);
  }

  // Plot solution
  plot(u[0]);

  return 0;
}

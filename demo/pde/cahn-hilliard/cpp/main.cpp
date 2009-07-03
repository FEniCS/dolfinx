// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-03-02
// Last changed: 2008-12-27
//
// This program illustrates the use of the DOLFIN nonlinear solver for solving
// the Cahn-Hilliard equation.
//
// The Cahn-Hilliard equation is very sensitive to the chosen parameters and
// time step. It also requires fines meshes, and is often not well-suited to
// iterative linear solvers.

#include <dolfin.h>
#include "CahnHilliard2D.h"
#include "CahnHilliard3D.h"

using namespace dolfin;

// Initial conditions
class InitialConditions: public Function
{
public:

  InitialConditions(boost::shared_ptr<const FunctionSpace> V) : Function(V)
  {
    dolfin::seed(2);
  }

  void eval(double* values, const Data& data) const
  {
    values[0]= 0.0; 
    values[1]= 0.63 + 0.02*(0.5 - dolfin::rand()); 
  }

};

// User defined nonlinear problem
class CahnHilliardEquation : public NonlinearProblem, public Parametrized
{
  public:

    // Constructor
    CahnHilliardEquation(const Mesh& mesh, Function& u, Function& u0, Constant& dt,
                         Constant& theta, Constant& lambda, Constant& mu_factor)
                       : a(0), L(0), reset_Jacobian(true)
    {
      // Create forms
      if (mesh.topology().dim() == 2)
      {
        CahnHilliard2D::CoefficientSet coeffs;
        coeffs.u = u; coeffs.u0 = u0;
        coeffs.lmbda = lambda; coeffs.muFactor = mu_factor;
        coeffs.dt = dt; coeffs.theta = theta;

        boost::shared_ptr<CahnHilliard2D::FunctionSpace> V(new CahnHilliard2D::FunctionSpace(mesh));
        a = new CahnHilliard2D::BilinearForm(V, V, coeffs);
        L = new CahnHilliard2D::LinearForm(V, coeffs);

        // Set solution to intitial condition
        InitialConditions u_initial(V);
        u = u_initial;
      }
      else if (mesh.topology().dim() == 3)
      {
        CahnHilliard3D::CoefficientSet coeffs;
        coeffs.u = u; coeffs.u0 = u0;
        coeffs.lmbda = lambda; coeffs.muFactor = mu_factor;
        coeffs.dt = dt; coeffs.theta = theta;

        boost::shared_ptr<CahnHilliard3D::FunctionSpace> V(new CahnHilliard3D::FunctionSpace(mesh));
        a = new CahnHilliard3D::BilinearForm(V, V, coeffs);
        L = new CahnHilliard3D::LinearForm(V, coeffs);

        // Set solution to intitial condition
        InitialConditions u_initial(V);
        u = u_initial;
      }
      else
        error("Cahn-Hilliard model is programmed for 2D and 3D only");
    }

    // Destructor
    ~CahnHilliardEquation()
    {
      delete a;
      delete L;
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

  private:

    // Pointers to FunctionSpace and forms
    Form* a;
    Form* L;
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
  Constant mu_factor(100.0);

  double t = 0.0;
  double T = 50*dt;

  // Solution functions
  Function u;
  Function u0;

  // Create user-defined nonlinear problem
  CahnHilliardEquation cahn_hilliard(mesh, u, u0, dt, theta, lambda, mu_factor);

  // Create nonlinear solver and set parameters
  NewtonSolver newton_solver("lu");
  newton_solver.parameters("convergence_criterion") = "incremental";
  newton_solver.parameters("maximum_iterations") = 10;
  newton_solver.parameters("relative_tolerance") = 1e-6;
  newton_solver.parameters("absolute_tolerance") = 1e-15;

  // Save initial condition to file
  File file("cahn_hilliard.pvd");
  file << u[1];

  // Solve
  while (t < T)
  {
    // Update for next time step
    t += dt;
    u0.vector() = u.vector();

    // Solve
    newton_solver.solve(cahn_hilliard, u.vector());

    // Save function to file
    file << u[1];
  }

  // Plot solution
  plot(u[1]);

  return 0;
}

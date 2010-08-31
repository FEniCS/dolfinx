// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-03-02
// Last changed: 2010-01-31
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
class InitialConditions : public Expression
{
public:

  InitialConditions(const Mesh& mesh) : Expression(mesh.topology().dim())
  {
    dolfin::seed(2);
  }

  void eval(Array<double>& values, const Data& data) const
  {
    values[0]= 0.0;
    values[1]= 0.63 + 0.02*(0.5 - dolfin::rand());
  }

};

// User defined nonlinear problem
class CahnHilliardEquation : public NonlinearProblem
{
  public:

    // Constructor
    CahnHilliardEquation(const Mesh& mesh, Constant& dt,
                         Constant& theta, Constant& lambda, Constant& mu_factor)
                       : a(0), L(0), _u(0), _u0(0), reset_Jacobian(true)
    {
      if (mesh.topology().dim() == 2)
      {
        // Create forms
        boost::shared_ptr<CahnHilliard2D::FunctionSpace> V(new CahnHilliard2D::FunctionSpace(mesh));
        a = new CahnHilliard2D::BilinearForm(V, V);
        L = new CahnHilliard2D::LinearForm(V);

        _u  = new Function(V);
        _u0 = new Function(V);

        // Attach coefficients
        CahnHilliard2D::BilinearForm* _a = dynamic_cast<CahnHilliard2D::BilinearForm*>(a);
        CahnHilliard2D::LinearForm*   _L = dynamic_cast<CahnHilliard2D::LinearForm*>(L);
        _a->u = *_u;
        _a->lmbda = lambda; _a->muFactor = mu_factor;
        _a->dt = dt; _a->theta = theta;
        _L->u = *_u; _L->u0 = *_u0;
        _L->lmbda = lambda; _L->muFactor = mu_factor;
        _L->dt = dt; _L->theta = theta;

        // Set solution to intitial condition
        InitialConditions u_initial(mesh);
        *_u = u_initial;
      }
      else if (mesh.topology().dim() == 3)
      {
        boost::shared_ptr<CahnHilliard3D::FunctionSpace> V(new CahnHilliard3D::FunctionSpace(mesh));
        a = new CahnHilliard3D::BilinearForm(V, V);
        L = new CahnHilliard3D::LinearForm(V);

        _u  = new Function(V);
        _u0 = new Function(V);

        // Attach coefficients
        CahnHilliard3D::BilinearForm* _a = dynamic_cast<CahnHilliard3D::BilinearForm*>(a);
        CahnHilliard3D::LinearForm*   _L = dynamic_cast<CahnHilliard3D::LinearForm*>(L);
        _a->u = *_u;
        _a->lmbda = lambda; _a->muFactor = mu_factor;
        _a->dt = dt; _a->theta = theta;
        _L->u = *_u; _L->u0 = *_u0;
        _L->lmbda = lambda; _L->muFactor = mu_factor;
        _L->dt = dt; _L->theta = theta;

        // Set solution to intitial condition
        InitialConditions u_initial(mesh);
        *_u = u_initial;
      }
      else
        error("Cahn-Hilliard model is programmed for 2D and 3D only");
    }

    // Destructor
    ~CahnHilliardEquation()
    {
      delete a;
      delete L;
      delete _u;
      delete _u0;
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
    Form* a;
    Form* L;
    Function* _u;
    Function* _u0;
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

  // Create user-defined nonlinear problem
  CahnHilliardEquation cahn_hilliard(mesh, dt, theta, lambda, mu_factor);

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

// Copyright (C) 2005-2006 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-03-02
// Last changed: 2006-03-02
//
// This program illustrates the use of the DOLFIN nonlinear solver for solving 
// the Cahn-Hilliard equation.
//
// The Cahn-Hilliard equation is very sensitive to the chosen parameters and
// time step. It also requires fines meshes, and is often not well-suited to
// iterative linear solvers.
//

#include <dolfin.h>
#include "CahnHilliard2D.h"
#include "CahnHilliard3D.h"

using namespace dolfin;

// User defined nonlinear problem 
class CahnHilliardEquation : public NonlinearProblem, public Parametrized
{
  public:

    // Constructor 
    CahnHilliardEquation(Mesh& mesh, Function& U, Function& rate1, Function& U0, 
        Function& rate0, real& theta, real& dt) : NonlinearProblem(), 
        Parametrized(), _mesh(&mesh), _dt(&dt), _theta(&theta)
    {

      lambda = 5.0e-2;
      add("mobility", "nonlinear");
      add("chemical potential", "logarithmic");

      // Create forms
      if(mesh.numSpaceDim() == 2)
      {
        a = new CahnHilliard2D::BilinearForm(U, lambda, *_dt, *_theta);
        L = new CahnHilliard2D::LinearForm(U, U0, rate0, lambda, *_dt, *_theta);
      }
      else if(mesh.numSpaceDim() == 3)
      {
        a = new CahnHilliard3D::BilinearForm(U, lambda, *_dt, *_theta);
        L = new CahnHilliard3D::LinearForm(U, U0, rate0, lambda, *_dt, *_theta);
      }
      else
        dolfin_error("Cahn-Hilliard model is programmed for 2D and 3D only");

      // Initialise solution functions
      U.init(mesh, a->trial());
      rate1.init(mesh, a->trial());
      U0.init(mesh, a->trial());
      rate0.init(mesh, a->trial());
    }

    // Destructor 
    ~CahnHilliardEquation()
    {
      delete a; 
      delete L;
    }
 
    // User defined assemble of Jacobian and residual vector 
    void form(GenericMatrix& A, GenericVector& b, const GenericVector& x)
    {
      // Assemble system and RHS (Neumann boundary conditions)
      dolfin_log(false);
      FEM::assemble(*a, *L, A, b, *_mesh);
      dolfin_log(true);
    }

  private:

    // Pointers to forms and  mesh
    BilinearForm *a;
    LinearForm *L;
    Mesh* _mesh;

    // Mobility and its derivative 
    Function *mobility, *d_mobility;

    // Chemical potential and its derivative 
    Function *mu, *dmu_dc;
 
    real *_dt, *_theta;
    // Surface parameter
    real lambda;
};


int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);

  // Set up problem
  UnitSquare mesh(40, 40);

  // Set time stepping parameters
  real dt = 2.0e-6; real t  = 0.0; real T  = 500*dt;
  real theta = 0.5;

  Function U, U0;
  Function rate1, rate0;

  // Create user-defined nonlinear problem
  CahnHilliardEquation cahn_hilliard(mesh, U, rate1, U0, rate0, theta, dt);

  // Create nonlinear solver and set parameters
  NewtonSolver newton_solver;
  newton_solver.set("Newton convergence criterion", "incremental");
  newton_solver.set("Newton maximum iterations", 10);
  newton_solver.set("Newton relative tolerance", 1e-6);
  newton_solver.set("Newton absolute tolerance", 1e-15);

  Vector& x = U.vector();

  // Randomly perturbed intitial conditions
  dolfin::uint size = FEM::size(mesh, U.element()[0]);
  dolfin::seed(2);
  for(dolfin::uint i=0; i < size; ++i)
     x(i) = 0.63 + 0.02*(0.5-dolfin::rand());

  // Save initial condition to file
  File file("cahn_hilliard.pvd");
  Function c = U[0];
  c = U[0];
  file << c;

  Vector& r  = rate1.vector();
  Vector& x0 = U0.vector();
  Vector& r0 = rate0.vector();

  while( t < T)
  {
    // Update for next time step
    t += dt;
    U0    = U;
    rate0 = rate1;

    // Solve
    newton_solver.solve(cahn_hilliard, x);

    // Compute rate
    r = r0;
    r *= -((1.0-theta)/theta);

    r += (1.0/(theta*dt))*x;
    r -= (1.0/(theta*dt))*x0;
//    r.axpy((1.0/(theta*dt)), x);
//    r.axpy(-(1.0/(theta*dt)), x0);

   // Save function to file
    c = U[0];
    file << c;
  }

  return 0;
}


// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-03-02
// Last changed: 2007-05-24
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

// User defined nonlinear problem 
class CahnHilliardEquation : public NonlinearProblem, public Parametrized
{
  public:

    // Constructor 
    CahnHilliardEquation(Mesh& mesh, Function& u, Function& u0, Function& dt, 
                         Function& theta, Function& lambda, Function& muFactor) 
         : assembler(mesh), reset_Jacobian(true)
    {
      // Create forms
      if(mesh.topology().dim() == 2)
      {
        a = new CahnHilliard2DBilinearForm(u, lambda, muFactor, dt, theta);
        L = new CahnHilliard2DLinearForm(u, u0, lambda, muFactor, dt, theta);
      }
      else if(mesh.topology().dim() == 3)
      {
        a = new CahnHilliard3DBilinearForm(u, lambda, muFactor, dt, theta);
        L = new CahnHilliard3DLinearForm(u, u0, lambda, muFactor, dt, theta);
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
 
    // Return forms 
    Form& form(dolfin::uint i) const
    {
      if( i == 1)
        return *L;
      else if( i == 2)
        return *a;
      else
        error("Can only return linear or bilinear form.");
      return *L;
    }

    // User defined assemble of Jacobian and residual vector 
    void form(GenericMatrix& A, GenericVector& b, const GenericVector& x)
    {
      // Assemble system and RHS (Neumann boundary conditions)
      assembler.assemble(A, *a, reset_Jacobian);
      reset_Jacobian  = false;
      assembler.assemble(b, *L);
    
    }

  private:

    // Pointers to forms
    Form *a;
    Form *L;

    Assembler assembler;
    bool reset_Jacobian;
};

int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);

  // Mesh
  UnitSquare mesh(80, 80);

  // Time stepping and model parameters
  real delta_t = 1.0e-5;
  Function dt(mesh, delta_t); 
  Function theta(mesh, 0.5); 
  Function lambda(mesh, 1.0e-2); 
  Function muFactor(mesh, 100.0); 

  real t = 0.0; 
  real T = 50*delta_t;

  // Solution functions
  Function u; 
  Function u0;

  // Create user-defined nonlinear problem
  CahnHilliardEquation cahn_hilliard(mesh, u, u0, dt, theta, lambda, muFactor);

  // Initialise discrete functions
  Vector x, x0;
  u.init(mesh,   x, cahn_hilliard.form(1), 1);
  u0.init(mesh, x0, cahn_hilliard.form(1), 1);

  // Create nonlinear solver and set parameters
  NewtonSolver newton_solver;
  newton_solver.set("Newton convergence criterion", "incremental");
  newton_solver.set("Newton maximum iterations", 10);
  newton_solver.set("Newton relative tolerance", 1e-6);
  newton_solver.set("Newton absolute tolerance", 1e-15);

  // Randomly perturbed intitial conditions
  dolfin::seed(2);
  dolfin::uint size = mesh.numVertices();
  real* x_init = new real[size];
  unsigned int* x_pos = new unsigned int[size];
  for(dolfin::uint i=0; i < size; ++i)
  {
     x_init[i] = 0.63 + 0.02*(0.5-dolfin::rand());
     x_pos[i]  = i + size;
  }
  x.set(x_init, size, x_pos);
  x.apply();
  delete [] x_init;
  delete [] x_pos;

  // Save initial condition to file
  File file("cahn_hilliard.pvd");
  Function c;
  c = u[1];
  file << c;

  while( t < T)
  {
    // Update for next time step
    t += delta_t;
    u0 = u;
    
    // Solve
    newton_solver.solve(cahn_hilliard, x);

    // Save function to file
    c = u[1];
    file << c;
  }

  // Plot solution
  plot(c);
  
  return 0;
}

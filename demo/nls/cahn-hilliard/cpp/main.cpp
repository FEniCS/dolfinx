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
    CahnHilliardEquation(Mesh& mesh, Function& u, Function& u0, Constant& dt, 
                         Constant& theta, Constant& lambda, Constant& muFactor) 
                       : reset_Jacobian(true)
    {
      // Create forms
      if(mesh.topology().dim() == 2)
      {
        V = new CahnHilliard2DFunctionSpace(mesh);
        a = new CahnHilliard2DBilinearForm(*V, *V);
        L = new CahnHilliard2DLinearForm(*V);
        a->w0 = u;
        a->lmbda = lambda;
        a->muFactor = muFactor;
        a->dt = dt;
        a->theta = theta;
        L->w0 = u;
        L->w1 = u0;
        L->lmbda = lambda;
        L->muFactor = muFactor;
        L->dt = dt;
        L->theta = theta;
      }
      else if(mesh.topology().dim() == 3)
      {
        error("3D Cahn-Hilliard demo need to be updated for new Function interface.");
        //V = new CahnHilliard3DFunctionSpace(mesh);
        //a = new CahnHilliard3DBilinearForm(*V, *V);
        //L = new CahnHilliard3DLinearForm(*V);
      }
      else
        error("Cahn-Hilliard model is programmed for 2D and 3D only");
    }

    // Destructor 
    ~CahnHilliardEquation()
    {
      delete V; 
      delete a; 
      delete L;
    }
 
    // User defined residual vector 
    void F(GenericVector& b, const GenericVector& x)
    {
      // Assemble RHS (Neumann boundary conditions)
      Assembler::assemble(b, *L);
    }

    // User defined assemble of Jacobian 
    void J(GenericMatrix& A, const GenericVector& x)
    {
      // Assemble system and RHS (Neumann boundary conditions)
      Assembler::assemble(A, *a, reset_Jacobian);
      reset_Jacobian  = false;
    }

  private:

    // Pointers to FunctionSpace and forms
    FunctionSpace* V;
    CahnHilliard2DBilinearForm* a;
    CahnHilliard2DLinearForm* L;

    bool reset_Jacobian;
};

int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);

  // Mesh
  UnitSquare mesh(80, 80);

  // Time stepping and model parameters
  double delta_t = 5.0e-6;
  Constant dt(delta_t); 
  Constant theta(0.5); 
  Constant lambda(1.0e-2); 
  Constant muFactor(100.0); 

  double t = 0.0; 
  double T = 50*delta_t;

  // Solution functions
  Function u; 
  Function u0;

  // Create user-defined nonlinear problem
  CahnHilliardEquation cahn_hilliard(mesh, u, u0, dt, theta, lambda, muFactor);

  // Create nonlinear solver and set parameters
  //NewtonSolver newton_solver;
  NewtonSolver newton_solver(lu);
  newton_solver.set("Newton convergence criterion", "incremental");
  newton_solver.set("Newton maximum iterations", 10);
  newton_solver.set("Newton relative tolerance", 1e-6);
  newton_solver.set("Newton absolute tolerance", 1e-15);

  // Randomly perturbed intitial conditions
  dolfin::seed(2);
  dolfin::uint size = mesh.numVertices();
  double* x_init = new double[size];
  unsigned int* x_pos = new unsigned int[size];
  for(dolfin::uint i=0; i < size; ++i)
  {
     x_init[i] = 0.63 + 0.02*(0.5-dolfin::rand());
     x_pos[i]  = i + size;
  }
  u.vector().set(x_init, size, x_pos);
  u.vector().apply();
  delete [] x_init;
  delete [] x_pos;

  // Save initial condition to file
  File file("cahn_hilliard.pvd");
  file << u[1];

  while( t < T)
  {
    // Update for next time step
    t += delta_t;
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

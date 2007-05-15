// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-03-02
// Last changed: 2007-05-15
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
    CahnHilliardEquation(Mesh& mesh, Function& u, Function& u0, Function& dt, 
                         Function& theta, Function& lambda, Function& muFactor) 
                         : _mesh(&mesh), _dt(&dt), _theta(&theta), 
                           _lambda(&lambda), _muFactor(&muFactor)
    {
      // Create forms
      if(mesh.topology().dim() == 2)
      {
        cout << "Create forms " << endl;
        a = new CahnHilliard2DBilinearForm(u, *_lambda, *_muFactor, *_dt, *_theta);
        L = new CahnHilliard2DLinearForm(u, u0, *_lambda, *_muFactor, *_dt, *_theta);
      }
      else if(mesh.topology().dim() == 3)
      {
        a = new CahnHilliard3DBilinearForm(u, *_lambda, *_muFactor, *_dt, *_theta);
        L = new CahnHilliard3DLinearForm(u, u0, *_lambda, *_muFactor, *_dt, *_theta);
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
    const Form& form(dolfin::uint i) const
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
      Assembler assembler;
      assembler.assemble(A, *a, *_mesh);
      assembler.assemble(b, *L, *_mesh);
    }

  private:

    // Pointers to forms and mesh
    Form *a;
    Form *L;
    Mesh* _mesh;

    // Time stepping parameters
    Function* _dt; 
    Function* _theta;

    // Model parameters
    Function* _lambda; 
    Function* _muFactor;
};



int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);

  // Mesh
  UnitSquare mesh(80, 80);

  // Time stepping and model parameters
  real delta_t = 2.0e-6;
  Function dt(mesh, delta_t); 
  Function theta(mesh, 0.5); 
  Function lambda(mesh, 1.0e-2); 
  Function muFactor(mesh, 100.0); 

  real t  = 0.0; 
  real T  = 50*delta_t;

  // Solution functions
  Function u; 
  Function u0;

  // Create user-defined nonlinear problem
  CahnHilliardEquation cahn_hilliard(mesh, u, u0, dt, theta, lambda, muFactor);

  // Initialise discrete functions
  cout << "Initialise functions " << endl;
  Vector x, x0;
  u.init(mesh,   x, cahn_hilliard.form(1), 1);
  u0.init(mesh, x0, cahn_hilliard.form(1), 1);
  cout << "Finsihed initialise functions " << endl;

  // Create nonlinear solver and set parameters
  NewtonSolver newton_solver;
  newton_solver.set("Newton convergence criterion", "incremental");
  newton_solver.set("Newton maximum iterations", 10);
  newton_solver.set("Newton relative tolerance", 1e-6);
  newton_solver.set("Newton absolute tolerance", 1e-15);

  // Randomly perturbed intitial conditions
  dolfin::uint size = mesh.numVertices();
  dolfin::seed(2);
  for(dolfin::uint i=size; i < 2*size; ++i)
     x(i) = 0.63 + 0.02*(0.5-dolfin::rand());

  // Save initial condition to file
  File file("cahn_hilliard.pvd");
  Function c;
  c = u[1];
  file << c;

  File file_u("u.xml");
  File file_u0("u0.xml");
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

  return 0;
}


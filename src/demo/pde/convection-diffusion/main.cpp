// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-02-09
// Last changed: 2006-03-24
//
// This demo first computes the flow around a dolphin by solving
// the Stokes equations using a mixed formulation with Taylor-Hood
// elements. The temperature around the dolphin is then computed
// by solving the time-dependent convection-diffusion equation by
// a least-squares stabilized cG(1)cG(1) method.

#include <dolfin.h>
#include "Stokes.h"
#include "ConvectionDiffusion.h"

using namespace dolfin;

void solveConvectionDiffusion(Mesh& mesh, Function& velocity)
{
  // Boundary condition
  class MyBC : public BoundaryCondition
  {
    void eval(BoundaryValue& value, const Point& p, unsigned int i)
    {
      if ( p.x == 1.0 )
	value = 1.0;
      else if ( p.x != 0.0 && p.x != 1.0 && p.y != 0.0 && p.y != 1.0 )
	value = 1.0;
    }
  };

  MyBC bc;

  // Linear system and solver
  Matrix A;
  Vector x, b;
  LU solver;
  
  // Create functions
  ConvectionDiffusion::BilinearForm::TrialElement element;
  Function U0 = 0.0;
  Function U1(x, mesh, element);

  // Create forms
  Function f = 0.0;
  ConvectionDiffusion::BilinearForm a(velocity);
  ConvectionDiffusion::LinearForm L(U0, velocity, f);

  // Assemble left-hand side
  FEM::assemble(a, A, mesh);
  
  // Parameters for time-stepping
  real T = 2.0;
  real k = 0.05;
  real t = k;
  
  // Output file
  File file("temperature.pvd");

  // Start time-stepping
  Progress p("Time-stepping");
  while ( t < T )
  {
    // Assemble load vector and set boundary conditions
    FEM::assemble(L, b, mesh);
    FEM::applyBC(A, b, mesh, element, bc);
    
    // Solve the linear system
    solver.solve(A, x, b);
    
    // Save the solution to file
    file << U1;

    // Update progress
    p = t / T;

    // Move to next interval
    t += k;
    U0 = U1;
  }
}

int main()
{
  // Boundary condition
  class MyBC : public BoundaryCondition
  {
    void eval(BoundaryValue& value, const Point& p, unsigned int i)
    {
      // Pressure boundary condition, zero pressure at one point
      if ( i == 2 )
      {
        if ( p.x < DOLFIN_EPS && p.y < DOLFIN_EPS )
        {
          value = 0.0;
        }
        return;
      }
      
      // Velocity boundary condition at inflow
      if ( p.x > (1.0 - DOLFIN_EPS) )
      {
        if ( i == 0 )
          value = -1.0;
        else
          value = 0.0;
        return;
      }
      
      // Velocity boundary condition at remaining boundary (excluding outflow)
      if ( p.x > DOLFIN_EPS )
        value = 0.0;
    }
  };

  MyBC bc;

  // Set up problem
  Mesh mesh("dolfin-2.xml.gz");
  Function f = 0.0;
  Stokes::BilinearForm a;
  Stokes::LinearForm L(f);
  PDE pde(a, L, mesh, bc);

  // Compute solution
  Function U;
  Function P;
  set("Krylov shift nonzero", 1e-10);  
  pde.solve(U, P);

  // Save solution to file
  File ufile("velocity.pvd");
  File pfile("pressure.pvd");
  ufile << U;
  pfile << P;

  // Solve convection-diffusion with computed velocity field
  set("Krylov shift nonzero", 0.0);
  solveConvectionDiffusion(mesh, U);
}

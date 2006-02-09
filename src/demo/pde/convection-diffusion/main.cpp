// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// This demo first computes the flow around a dolphin by solving
// the Stokes equations using a mixed formulation with Taylor-Hood
// elements. The temperature around the dolphin is then computed
// by solving the time-dependent convection-diffusion equation by
// a least-squares stabilized cG(1)cG(1) method.

// First added:  2006-02-09
// Last changed: 2006-02-09

#include <dolfin.h>
#include "Stokes.h"
#include "ConvectionDiffusion.h"

using namespace dolfin;

void solveConvDiff(Mesh& mesh, Function& b)
{
  // Boundary condition
  class BC : public BoundaryCondition
  {
    void eval(BoundaryValue& value, const Point& p, unsigned int i)
    {
      if ( p.x == 1.0 )
	value = 0.0;
      else if ( p.x != 0.0 && p.x != 1.0 && p.y != 0.0 && p.y != 1.0 )
	value = 1.0;
    }
  };

  // Stabilization
  class Stabilization : public Function
  {
    real eval(const Point& p, unsigned int i)
    {
      return 0.0;
    }
  };

  // Setup
  File file("temperature.pvd");
  GMRES solver;
  Stabilization delta;
  Zero f;
  BC bc;
  Vector x0, x1, r;
  Function u0(x0, mesh);

  // Create forms
  ConvectionDiffusion::BilinearForm a(b, delta);
  ConvectionDiffusion::LinearForm L(u0, b, f, delta);
  
  // Initialize x1
  x1.init(FEM::size(mesh, a.trial()));
  Function u1(x1, mesh, a.trial());

  // Assemble left-hand side
  Matrix A;
  FEM::assemble(a, A, mesh);
  
  // Parameters for time-stepping
  real t = 0.0;
  real T = 0.3;
  real k = 0.01;

  // Start time-stepping
  Progress p("Time-stepping");
  while ( t < T )
  {
    // Make time step
    t += k;
    x0 = x1;
    
    // Assemble load vector and set boundary conditions
    FEM::assemble(L, r, mesh);
    FEM::applyBC(A, r, mesh, a.trial(), bc);
    
    // Solve the linear system
    solver.solve(A, x1, r);
    
    // Save the solution to file
    file << u1;

    // Update progress
    p = t / T;
  }
}

int main()
{
  // Boundary condition
  class BC : public BoundaryCondition
  {
    void eval(BoundaryValue& value, const Point& p, unsigned int i)
    {
      // Pressure boundary condition, zero pressure at one point
      if ( i == 2 && p.x < DOLFIN_EPS && p.y < DOLFIN_EPS )
      {
	value = 0.0;
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
  
  // Setup
  Mesh mesh("dolfin-2.xml.gz");
  Zero f;
  BC bc;

  // Create forms
  Stokes::BilinearForm a;
  Stokes::LinearForm L(f);

  // Assemble linear system
  Matrix A;
  Vector x, b;
  FEM::assemble(a, L, A, b, mesh, bc);

  // Solve the linear system
  GMRES solver;
  solver.solve(A, x, b);

  // Pick the two sub functions of the solution
  Function w(x, mesh, a.trial());
  Function u = w[0];
  Function p = w[1];

  // Save the solutions to file
  File ufile("velocity.pvd");
  File pfile("pressure.pvd");
  ufile << u;
  pfile << p;

  // Solve convection-diffusion with compute velocity field
  solveConvDiff(mesh, u);
}



/*
//-----------------------------------------------------------------------------
void ConvectionDiffusionSolver::ConvectionNormInv(Function& w, Vector& wnorm_vector, uint nsd)
{
  real tau = 1.0;  // stabilisation parameter

  real norm;
  wnorm_vector.init(mesh.noVertices());
  for (VertexIterator vertex(mesh); !vertex.end(); ++vertex)
  {
    norm = 0.0;
    for (uint i =0; i < nsd; ++i)
      norm += w(*vertex, i)*w(*vertex, i);

    norm = 0.5*tau/sqrt(norm);
    wnorm_vector((*vertex).id()) = norm;  
  }

}
//-----------------------------------------------------------------------------

*/

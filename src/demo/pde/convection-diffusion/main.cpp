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

// Right-hand side for Stokes
class StokesForce : public Function
{
  real eval(const Point& p, unsigned int i)
  {
    return 0.0;
  }
};

// Boundary condition for Stokes
class StokesBC : public BoundaryCondition
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

// Boundary condition for convection-diffusion
class ConvDiffBC : public BoundaryCondition
{
  void eval(BoundaryValue& value, const Point& p, unsigned int i)
  {
    

  }
};

void solveConvDiff(Mesh& mesh, Function& b)
{
  /*

  // Right-hand side for convection-diffusion
  class ConvDiffSource : public Function
  {
    real eval(const Point& p, unsigned int i)
    {
      return 0.0;
    }
  };
  
  

   // Set up problem
  Mesh mesh("dolfin-2.xml.gz");
  ConvectionDiffusionSource f;
  ConvectionDiffusionBC bc;
  ConvectionDiffusionStabilization delta(b);
  ConvectionDiffusion::BilinearForm a;
  ConvectionDiffusion::LinearForm L(f);

  real t   = 0.0;  // current time

  Matrix A;              // matrix defining linear system
  Vector x0, x1, b;      // vectors 
  KrylovSolver solver(KrylovSolver::bicgstab); // linear solver
  Function u0(x0, mesh); // function at previous time step


  // Vectors for functions for element size and inverse of velocity norm
  Vector hvector, wnorm_vector; 

  // Functions for element size and inverse of velocity norm
  Function h(hvector), wnorm(wnorm_vector);

  // Create variational forms
  BilinearForm* a =0;
  LinearForm* L =0;
  if( nsd == 2 )
  {  
    a = new ConvectionDiffusion2D::BilinearForm(w, wnorm, h, k, c);
    L = new ConvectionDiffusion2D::LinearForm(u0, w, wnorm, f, h, k, c);
  } 
  else if ( nsd == 3 )
  {
    dolfin_error("3D convection-diffusion is currently disabled to limit compile time.");
//    a = new ConvectionDiffusion3D::BilinearForm(w, wnorm, h, k, c);
//    L = new ConvectionDiffusion3D::LinearForm(u0, w, wnorm, f, h, k, c);
  }
  else
  {
    dolfin_error("Convection-diffusion solver only implemented for 2 and 3 space dimensions.");
  }

  // Compute stabiliation term  tau/2|a|
  // It is assumed that the advective velocity can be prepresnted using a linear basis
  ConvectionNormInv(w, wnorm_vector, nsd);

  // Assemble stiffness matrix
  FEM::assemble(*a, A, mesh);

  uint N = FEM::size(mesh, a->trial());
  x1.init(N);
  Function u1(x1, mesh, a->trial());
  
  // Synchronize function u1 with time t
  u1.sync(t);

  // Start time-stepping
  Progress p("Time-stepping");
  while ( t < T )
  {
    // Make time step
    t += k;
    x0 = x1;
    
    // Assemble load vector and set boundary conditions
    FEM::assemble(*L, b, mesh);
    FEM::applyBC(A, b, mesh, a->trial(), bc);
    
    // Solve the linear system
    solver.solve(A, x1, b);
    
    // Save the solution to file
    file << u1;

    // Update progress
    p = t / T;
  }
  */
}

int main()
{
  // Set up problem
  Mesh mesh("dolfin-2.xml.gz");
  StokesForce f;
  StokesBC bc;
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

// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2005
//
// First added:  2003
// Last changed: 2005-12-31

#include "dolfin/ConvectionDiffusionSolver.h"
#include "dolfin/ConvectionDiffusion2D.h"
//#include "dolfin/ConvectionDiffusion3D.h"

using namespace dolfin;

void ConvectionDiffusionSolver::solve()
{
}
//-----------------------------------------------------------------------------
void ConvectionDiffusionSolver::solve(Mesh& mesh, Function& w, Function& f, 
                            BoundaryCondition& bc, real c, real k, real T)
{
  ConvectionDiffusionSolver solver(mesh, w, f, bc, c, k, T);
  solver.solve();
}
//-----------------------------------------------------------------------------
void ConvectionDiffusionSolver::ComputeElementSize(Mesh& mesh, Vector& h)
{
  // Compute element size h
  h.init(mesh.noCells());	
	for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    h((*cell).id()) = (*cell).diameter();
  }
}
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

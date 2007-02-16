// Copyright (C) 2005 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2005
// Last changed: 2006-05-07

#ifndef __NSE_SOLVER_H
#define __NSE_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin
{
  // This is a solver for the time dependent incompressible 
  // Navier-Stokes equations. 

  class NSESolver 
  {
  public:
    
    // Create the Navier-Stokes solver
    NSESolver(Mesh& mesh, Function& f, BoundaryCondition& bc_mom, 
	      BoundaryCondition& bc_con);
    
    // Solve Navier-Stokes equations
    void solve();

    // Solve Navier-Stokes equations (static version)
    static void solve(Mesh& mesh, Function& f, BoundaryCondition& bc_mom, 
		      BoundaryCondition& bc_con);

    // Compute cell diameter
    static void ComputeCellSize(Mesh& mesh, Vector& hvector);
      
    // Get minimum cell diameter
    static void GetMinimumCellSize(Mesh& mesh, real& hmin);

    // Compute stabilization 
    static void ComputeStabilization(Mesh& mesh, Function& w, real nu, real k, 
				     Vector& d1vector, Vector& d2vector);
    
    // Set initial velocity 
    void SetInitialVelocity(Vector& xvel);

  private:

    Mesh& mesh;
    Function& f;
    BoundaryCondition& bc_mom;
    BoundaryCondition& bc_con;

  };

}

#endif

// Copyright (C) 2005 Johan Hoffman.
// LiceALEd under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2005
// Last changed: 2006-05-07

#ifndef __ALE_SOLVER_H
#define __ALE_SOLVER_H

#include <dolfin/Solver.h>
#include <dolfin/ALEBoundaryCondition.h>
#include <dolfin/ALEFunction.h>
#include <dolfin/BoundaryMesh.h>


namespace dolfin
{
  /// This is a solver for the time dependent incompressible 
  /// Navier-Stokes equations using ALE formulation. 

  class ALESolver 
  {
  public:
    
    /// Create the Navier-Stokes ALE solver
    ALESolver(Mesh& mesh, Function& f, ALEBoundaryCondition& bc_mom, 
	      ALEBoundaryCondition& bc_con, ALEFunction& e);
   
    /// Solve Navier-Stokes ALE equations
    void solve();

    /// Solve Navier-Stokes ALE equations (static version)
    static void solve(Mesh& mesh, Function& f, ALEBoundaryCondition& bc_mom, 
		      ALEBoundaryCondition& bc_con, ALEFunction& e);

    /// Compute cell diameter
    void ComputeCellSize(Mesh& mesh, Vector& hvector);
      
    /// Get minimum cell diameter
    void GetMinimumCellSize(Mesh& mesh, real& hmin);

    /// Compute stabilization 
    void ComputeStabilization(Mesh& mesh, Function& w, real nu, real k, 
			      Vector& d1vector, Vector& d2vector);
    
    /// Set initial velocity 
    void SetInitialVelocity(Vector& xvel);

    
  private:

    Mesh&                 mesh;
    Function&             f;
    ALEBoundaryCondition& bc_mom;
    ALEBoundaryCondition& bc_con;
    ALEFunction&          e;

   

  };

}

#endif

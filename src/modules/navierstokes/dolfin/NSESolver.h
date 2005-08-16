// Copyright (C) 2005 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2005

#ifndef __NSE_SOLVER_H
#define __NSE_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin
{

  class NSESolver 
  {
  public:
    
    // Create Navier-Stokes solver
    NSESolver(Mesh& mesh, Function& f, BoundaryCondition& bc_mom, 
	      BoundaryCondition& bc_con);
    
    // Solve Navier-Stokes equations
    void solve();

    // Solve Navier-Stokes equations (static version)
    static void solve(Mesh& mesh, Function& f, BoundaryCondition& bc_mom, 
		      BoundaryCondition& bc_con);

    // Compute element diameter
    void ComputeElementSize(Mesh& mesh, Vector& hvector);
      
    // Compute inverse of norm of convection
    void ConvectionNormInv(Function& w, Function& wnorm,
			   Vector& wnorm_vector);

  private:

    Mesh& mesh;
    Function& f;
    BoundaryCondition& bc_mom;
    BoundaryCondition& bc_con;

  };

}

#endif

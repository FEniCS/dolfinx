// Copyright (C) 2003-2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
//  Modified by Garth N. Wells, 2005
//
// First added:  2003
// Last changed: 2006-05-07

#ifndef __CONVECTION_DIFFUSION_SOLVER_H
#define __CONVECTION_DIFFUSION_SOLVER_H

#ifdef HAVE_PETSC_H

#include <dolfin/Solver.h>

namespace dolfin
{

  class ConvectionDiffusionSolver : public Solver
  {
  public:
    
    // Create convection-diffusion solver
    ConvectionDiffusionSolver(Mesh& mesh, Function& w, Function& f, BoundaryCondition& bc,
                              real c, real k, real T);
    
    // Solve convection-diffusion
    void solve();
    
    // Solve convection-diffusion (static version)
    static void solve(Mesh& mesh, Function& w, Function& f, BoundaryCondition& bc,
                      real c, real k, real T);
  
    void ComputeElementSize(Mesh& mesh, Vector& h);

    void ConvectionNormInv(Function& w, Vector& wnorm_vector, uint nsd);

  private:

		
		Mesh& mesh;
    Function& w;
    Function& f;
    BoundaryCondition& bc;
    
    // diffusion
    real c;
    
    // dt and final time
    real k, T;

  };

}

#endif

#endif

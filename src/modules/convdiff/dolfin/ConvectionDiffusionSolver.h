// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#ifndef __CONVECTION_DIFFUSION_SOLVER_H
#define __CONVECTION_DIFFUSION_SOLVER_H

#include <dolfin/NewSolver.h>

namespace dolfin
{

  class ConvectionDiffusionSolver : public NewSolver
  {
  public:
    
    // Create convection-diffusion solver
    ConvectionDiffusionSolver(Mesh& mesh, NewFunction& w, NewFunction& f, NewBoundaryCondition& bc);
    
    // Solve convection-diffusion
    void solve();
    
    // Solve convection-diffusion (static version)
    static void solve(Mesh& mesh, NewFunction& w, NewFunction& f, NewBoundaryCondition& bc);
  
  private:

    Mesh& mesh;
    NewFunction& w;
    NewFunction& f;
    NewBoundaryCondition& bc;

  };

}

#endif

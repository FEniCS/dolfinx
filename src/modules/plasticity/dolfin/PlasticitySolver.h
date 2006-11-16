// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#ifndef __PLASTICITY_SOLVER_H
#define __PLASTICITY_SOLVER_H

#include <dolfin/Solver.h>
#include <dolfin/uBlasDenseMatrix.h>
#include <dolfin/uBlasVector.h>
#include <dolfin/PlasticityModel.h>

namespace dolfin
{
  
  class PlasticitySolver : public Solver
  {
  public:
    
    /// Constructor
    PlasticitySolver(Mesh& mesh, BoundaryCondition& bc, Function& f, 
                     const real dt, const real T, PlasticityModel& plastic_model);
    
    /// Solve plasticity problem
    void solve();
    
    /// Solve plasticity (static version)
    static void solve(Mesh& mesh, BoundaryCondition& bc, Function& f, 
                      const real dt, const real T, PlasticityModel& plastic_model);
    
  private:

    Mesh& mesh;
    BoundaryCondition& bc;
    Function& f;
    real dt;
    real T;
    PlasticityModel& plasticity_model;
  };
}

#endif

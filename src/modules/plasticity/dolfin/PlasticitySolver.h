// Copyright (C) 2006 Kristian Oelgaard and Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-11-13

#ifndef __PLASTICITY_SOLVER_H
#define __PLASTICITY_SOLVER_H

#include <dolfin/Solver.h>
#include <dolfin/NewtonSolver.h>
#include <dolfin/PDE.h>
#include <dolfin/uBlasDenseMatrix.h>
#include <dolfin/uBlasVector.h>
#include <dolfin/PlasticityModel.h>

namespace dolfin
{
  
  class PlasticitySolver : public Solver
  {
  public:
    
    /// Constructor
    PlasticitySolver(Mesh& mesh,
         BoundaryCondition& bc, Function& f, real dt, real T, PlasticityModel& plastic_model);
    
    /// Solve plasticity
    void solve();
    
    /// Solve plasticity (static version)
    static void solve(Mesh& mesh,
          BoundaryCondition& bc, Function& f, real dt, real T, PlasticityModel& plastic_model);
    
  private:

    /// Class variables
    Mesh& mesh;
    BoundaryCondition& bc;
    Function& f;
    real dt;
    real T;
    PlasticityModel& plastic_model;
  };
}

#endif

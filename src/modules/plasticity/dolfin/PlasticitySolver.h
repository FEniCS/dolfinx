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

#include "PlasticityModel.h"

namespace dolfin
{
  
  class PlasticitySolver : public Solver
  {
  public:
    
    // Create plasticity solver
    PlasticitySolver(Mesh& mesh,
         BoundaryCondition& bc, Function& f, real E, real nu,
         real dt, real T, PlasticityModel& plas);
    
    // Solve plasticity
    void solve();
    
    // Solve plasticity (static version)
    static void solve(Mesh& mesh,
          BoundaryCondition& bc, Function& f, real E, real nu,
          real dt, real T, PlasticityModel& plas);
    
  private:

    // constitutive matrix
    uBlasDenseMatrix C_m(double &lam, double &mu);
    
    Mesh& mesh;
    BoundaryCondition& bc;
    Function& f;
    real E;
    real nu;
    real dt;
    real T;
    PlasticityModel& plas;
    std::string output_dir;
  };
}

#endif

// Copyright (C) 2005 Johan Hoffman.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2005
// Last changed: 2006-05-07

#ifndef __CNS_SOLVER_H
#define __CNS_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin
{
  // This is a solver for the time dependent compressible 
  // Euler equations. 

  class CNSSolver 
  {
  public:
    
    // Create the Compressible solver
    CNSSolver(Mesh& mesh, Function& f, Function& initialdata, BoundaryCondition& bc);
    
    // Solve Compressible equations
    void solve();

    // Solve Copressible equations (static version)
    static void solve(Mesh& mesh, Function& f, Function& initialdata, BoundaryCondition& bc);

    // Compute cell diameter
    void ComputeCellSize(Mesh& mesh, Vector& hvector);
      
    // Get minimum cell diameter
    void GetMinimumCellSize(Mesh& mesh, real& hmin);

    // Compute stabilization 
    void ComputeStabilization(Mesh& mesh, Function& w, real k, 
			      Vector& dvector);
    
    // Set initial velocity 
    void SetInitialData(Function& rho0, Function& m0, Function& e0);

    // Compute velosity and pressure
    void ComputeUP(Mesh& mesh, Function& w, Function& p, Function& u, real gamma, int nsdim);

    // Set initial data
    //void finterpolate(Function& f1, Function& f2, Mesh& mesh);

    // Pick up the components of the w
    void ComputeRME(Mesh& mesh, Function& w, Function& rho, Function& m, Function& e);

    // Compute the volume inverse of each element
    void ComputeVolInv(Mesh& mesh, Vector& vol_inv);

    // Compute the shock capturing constant nu:
    void ComputeNu(Mesh& mesh, Function& res_rho, Function& res_m, Function& res_e, 
			      Function& nu_rho, Function& nu_m, Function& nu_e);

  private:
    Mesh& mesh;
    Function& f;
    Function& initial;
    BoundaryCondition& bc;
   };

}

#endif

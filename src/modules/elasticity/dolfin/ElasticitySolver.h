// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2006.
//
// First added:  2005
// Last changed: 2006-05-07

#ifndef __ELASTICITY_SOLVER_H
#define __ELASTICITY_SOLVER_H

#ifdef HAVE_PETSC_H

#include <dolfin/Solver.h>

namespace dolfin
{
  
  class ElasticitySolver : public Solver
  {
  public:
    
    // Create elasticity solver
    ElasticitySolver(Mesh& mesh,
		     Function& f, Function& u0, Function& v0,
		     real E, real nu,
		     BoundaryCondition& bc, real k, real T);
    
    // Solve elasticity
    void solve();
    
    void save(Mesh& mesh, Function& u, Function& v, File &solutionfile);

    // Solve elasticity (static version)
    static void solve(Mesh& mesh,
		      Function& f, Function& u0, Function& v0,
		      real E, real nu,
		      BoundaryCondition& bc, real k, real T);
    
  private:
    
    Mesh& mesh;
    Function& f;
    Function& u0;
    Function& v0;
    real E;
    real nu;
    BoundaryCondition& bc;
    real k;
    real T;
    int counter;

  };
  
}

#endif

#endif

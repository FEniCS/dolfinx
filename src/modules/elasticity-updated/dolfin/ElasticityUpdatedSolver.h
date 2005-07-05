// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2005

#ifndef __ELASTICITYUPDATED_SOLVER_H
#define __ELASTICITYUPDATED_SOLVER_H

#include <dolfin/Solver.h>

namespace dolfin
{
  
  class ElasticityUpdatedSolver : public Solver
  {
  public:
    
    // Create ElasticityUpdated solver
    ElasticityUpdatedSolver(Mesh& mesh,
			    Function& f, Function& v0,
			    real E, real nu, real nuv,
			    BoundaryCondition& bc, real k, real T);
    
    // Solve ElasticityUpdated
    void solve();
    
    void save(Mesh& mesh, File &solutionfile);

    // Solve ElasticityUpdated (static version)
    static void solve(Mesh& mesh,
		      Function& f, Function& v0,
		      real E, real nu, real nuv,
		      BoundaryCondition& bc, real k, real T);
    
  private:
    
    Mesh& mesh;
    Function& f;
    Function& v0;
    real E;
    real nu;
    real nuv;
    BoundaryCondition& bc;
    real k;
    real T;
    int counter;

  };
  
}

#endif

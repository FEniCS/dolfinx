// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

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
		     Function& f, Function& u0, Function& v0,
		     real E, real nu,
		     BoundaryCondition& bc, real k, real T);
    
    // Solve ElasticityUpdated
    void solve();
    
    void save(Mesh& mesh, File &solutionfile);

    // Solve ElasticityUpdated (static version)
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

// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2005

#ifndef __ELASTICITYUPDATED_SOLVER_H
#define __ELASTICITYUPDATED_SOLVER_H

#include <dolfin/Solver.h>
#include "dolfin/ElasticityUpdated.h"
#include "dolfin/ElasticityUpdatedSigma.h"
#include "dolfin/ElasticityUpdatedMass.h"

namespace dolfin
{
  
  class ElasticityUpdatedSolver : public Solver
  {
  public:
    
    // Create ElasticityUpdated solver
    ElasticityUpdatedSolver(Mesh& mesh,
			    Function& f, Function& v0, Function& rho,
			    real E, real nu, real nuv, real nuplast,
			    BoundaryCondition& bc, real k, real T);
    
    // Initialize data
    void init();

    // Solve ElasticityUpdated
    void solve();

    // Make a time step
    void step();
    
    // Prepare time step
    virtual void preparestep();

    virtual void save(Mesh& mesh, File &solutionfile, real t);
    void condsave(Mesh& mesh, File &solutionfile, real t);

    // Solve ElasticityUpdated (static version)
    static void solve(Mesh& mesh,
		      Function& f, Function& v0, Function& rho,
		      real E, real nu, real nuv, real nuplast,
		      BoundaryCondition& bc, real k, real T);
    
    Mesh& mesh;
    Function& f;
    Function& v0;
    Function& rho;
    real E;
    real nu;
    real nuv;
    real nuplast;
    BoundaryCondition& bc;
    real k;
    real T;
    int counter;
    real lastsample;
    real lambda;
    real mu;
    real t;
    real rtol;
    int maxiters;
    bool do_plasticity;
    real yield;

    // Elements

    ElasticityUpdated::LinearForm::TestElement element1;
    ElasticityUpdatedSigma::LinearForm::TestElement element2;
    ElasticityUpdatedSigma::LinearForm::TestElement element3;

    Vector x1_0, x1_1, x2_0, x2_1, b, m, msigma, stepresidual;
    Vector xsigma0, xsigma1, xepsilon1, xsigmanorm, mesh0;
    Vector xtmp1, xtmp2, xsigmatmp1;
    Matrix Dummy;
    
    Function v1;
    Function u0;
    Function u1;
    Function sigma0;
    Function sigma1;
    Function epsilon1;
    Function sigmanorm;

    // Forms

    ElasticityUpdated::LinearForm Lv;
    ElasticityUpdatedSigma::LinearForm Lsigma;
  };
  
}

#endif

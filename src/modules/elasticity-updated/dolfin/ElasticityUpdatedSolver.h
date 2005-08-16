// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2005

#ifndef __ELASTICITYUPDATED_SOLVER_H
#define __ELASTICITYUPDATED_SOLVER_H

#include <dolfin/Solver.h>
#include "dolfin/ElasticityUpdated.h"
#include "dolfin/ElasticityUpdatedSigma0.h"
#include "dolfin/ElasticityUpdatedSigma1.h"
#include "dolfin/ElasticityUpdatedSigma2.h"
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

    // Solve ElasticityUpdated (static version)
    static void solve(Mesh& mesh,
		      Function& f, Function& v0, Function& rho,
		      real E, real nu, real nuv, real nuplast,
		      BoundaryCondition& bc, real k, real T);
    
  protected:
    
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
    ElasticityUpdatedSigma0::LinearForm::TestElement element2_0;
    ElasticityUpdatedSigma1::LinearForm::TestElement element2_1;
    ElasticityUpdatedSigma2::LinearForm::TestElement element2_2;

    Vector x1_0, x1_1, x2_0, x2_1, b, m, msigma, stepresidual;
    Vector xsigma0_0, xsigma0_1, xsigma1_0, xsigma1_1, xsigma2_0, xsigma2_1,
      xepsilon0_1, xepsilon1_1, xepsilon2_1, xsigmanorm;
    Vector xtmp1, xtmp2, xtmp0_1, xtmp1_1, xtmp2_1;
    
    Function v1;
    Function sigma0_1;
    Function sigma1_1;
    Function sigma2_1;
    Function sigma0_0;
    Function sigma1_0;
    Function sigma2_0;
    Function epsilon0_1;
    Function epsilon1_1;
    Function epsilon2_1;
    Function sigmanorm;

    // Forms

    ElasticityUpdated::LinearForm Lv;
    ElasticityUpdatedSigma0::LinearForm Lsigma0;
    ElasticityUpdatedSigma1::LinearForm Lsigma1;
    ElasticityUpdatedSigma2::LinearForm Lsigma2;
  };
  
}

#endif

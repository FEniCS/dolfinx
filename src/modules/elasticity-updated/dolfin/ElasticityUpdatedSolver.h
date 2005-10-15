// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2005

#ifndef __ELASTICITYUPDATED_SOLVER_H
#define __ELASTICITYUPDATED_SOLVER_H

#include <dolfin/ODE.h>
#include <dolfin/TimeStepper.h>
#include <dolfin/Solver.h>
#include "dolfin/ElasticityUpdated.h"
#include "dolfin/ElasticityUpdatedSigma.h"
#include "dolfin/ElasticityUpdatedMass.h"

namespace dolfin
{
  
  class ElasticityUpdatedODE;

  class ElasticityUpdatedSolver : public Solver
  {
  public:
    
    // Create ElasticityUpdated solver
    ElasticityUpdatedSolver(Mesh& mesh,
			    Function& f, Function& v0, Function& rho,
			    real& E, real& nu, real& nuv, real& nuplast,
			    BoundaryCondition& bc, real& k, real& T);
    
    // Initialize data
    void init();

    // Solve ElasticityUpdated
    void solve();

    // Make a time step
    void step();

    // Old implementation of time integration (kept for reference)
    void oldstep();

    // Compute f(u) in dot(u) = f(u)
    void fu();

    // Gather x1ode, x2ode and xsigmaode into dotu
    void gather();

    // Scatter dotu into x1_1, x2_1 and xsigma1
//     void scatter();
    
    // Prepare time step
    virtual void preparestep();

    // Prepare iteration
    virtual void prepareiteration();

    virtual void save(Mesh& mesh, File &solutionfile, real t);
    void condsave(Mesh& mesh, File &solutionfile, real t);

    // Solve ElasticityUpdated (static version)
    static void solve(Mesh& mesh,
		      Function& f, Function& v0, Function& rho,
		      real& E, real& nu, real& nuv, real& nuplast,
		      BoundaryCondition& bc, real& k, real& T);
    
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

    real savesamplefreq;

    int fevals;

    uint Nv, Nsigma, Nsigmanorm;

    // ODE

    ElasticityUpdatedODE* ode;
    TimeStepper* ts;

    // Elements

    ElasticityUpdated::LinearForm::TestElement element1;
    ElasticityUpdatedSigma::LinearForm::TestElement element2;
    ElasticityUpdatedSigma::LinearForm::FunctionElement_2 element3;

    Vector x1_0, x1_1, x2_0, x2_1, b, m, msigma, stepresidual;
    Vector xsigma0, xsigma1, xepsilon1, xsigmanorm, xjaumann1;
    Vector xtmp1, xtmp2, xsigmatmp1, xsigmatmp2;
    Vector fcontact;
    Matrix Dummy;

    Vector x1ode, x2ode, xsigmaode, dotu;

    VecScatter x1odesc, x2odesc, xsigmaodesc;
    IS x1odeis, x2odeis, xsigmaodeis;
    


    int* x1ode_indices;
    int* x2ode_indices;
    int* xsigmaode_indices;

    real* uode;
    real* yode;
    
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

  class ElasticityUpdatedODE : public ODE
  {
  public:
    ElasticityUpdatedODE(ElasticityUpdatedSolver& solver);
    real u0(unsigned int i);
    /// Evaluate right-hand side (mono-adaptive version)
    virtual void f(const real u[], real t, real y[]);
    virtual bool update(const real u[], real t, bool end);

    void fromArray(const real u[], Vector& x, uint offset, uint size);
    void toArray(real y[], Vector&x, uint offset, uint size);

    ElasticityUpdatedSolver& solver;
  };


  
}

#endif

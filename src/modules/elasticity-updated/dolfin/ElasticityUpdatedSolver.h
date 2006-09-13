// Copyright (C) 2005 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
// Modified by Anders Logg 2006.
//
// First added:  2005
// Last changed: 2006-08-21

#ifndef __ELASTICITYUPDATED_SOLVER_H
#define __ELASTICITYUPDATED_SOLVER_H

#ifdef HAVE_PETSC_H

#include <dolfin/Vector.h>
#include <dolfin/ODE.h>
#include <dolfin/TimeStepper.h>
#include <dolfin/Solver.h>


namespace dolfin
{
  
  class ElasticityUpdatedODE;


  class ElasticityUpdatedSolver : public Solver
  {
  public:
    
    // Create ElasticityUpdated solver
    ElasticityUpdatedSolver(Mesh& mesh,
			    Function& f, Function& v0, Function& rho,
			    real E, real nu, real nuv, real nuplast,
			    BoundaryCondition& bc, real k, real T);
    
    ElasticityUpdatedSolver& operator=(const ElasticityUpdatedSolver& solver);

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

    // Gather x1 (subvector) into x2
    static void gather(Vector& x1, Vector& x2, VecScatter& x1sc);

    // Scatter part of x2 into x1 (subvector)
    static void scatter(Vector& x1, Vector& x2, VecScatter& x1sc);

    // Create Scatterer
    static VecScatter* createScatterer(Vector& x1, Vector& x2, int offset,
				       int size);

    // Scatter x2 into x1 (subvector)
//     void scatter(Vector& x1, Vector& x2, VecScatter& x1sc);

    static void fromArray(const real u[], Vector& x, uint offset, uint size);
    static void toArray(real y[], Vector&x, uint offset, uint size);

    static void fromDense(const uBlasVector& u, Vector& x, uint offset,
			  uint size);
    static void toDense(uBlasVector& y, Vector&x, uint offset, uint size);

    // Prepare time step
    virtual void preparestep();

    // Prepare iteration
    virtual void prepareiteration();

    virtual void save(Mesh& mesh, File &solutionfile, real t);
    void condsave(Mesh& mesh, File &solutionfile, real t);

    // Solve ElasticityUpdated (static version)
    static void solve(Mesh& mesh,
		      Function& f, Function& v0, Function& rho,
		      real E, real nu, real nuv, real nuplast,
		      BoundaryCondition& bc, real k, real T);
    

    // Utility functions

    static void finterpolate(Function& f1, Function& f2, Mesh& mesh);

    static void plasticity(Vector& xsigma, Vector& xsigmanorm, real yield,
			   FiniteElement& element2, Mesh& mesh);

    static void initmsigma(Vector& msigma,
			   FiniteElement& element2, Mesh& mesh);

    static void initu0(Vector& x0,
		       FiniteElement& element, Mesh& mesh);


    static void initJ0(Vector& xJ0,
		       FiniteElement& element, Mesh& mesh);

    static void computeJ(Vector& xJ0, Vector& xJ, Vector& xJinv,
			 FiniteElement& element, Mesh& mesh);

    static void initF0Green(Vector& xF0,
			    FiniteElement& element1, Mesh& mesh);

    static void computeFGreen(Vector& xF, Vector& xF0, Vector& xF1,
			      FiniteElement& element1, Mesh& mesh);

    static void initF0Euler(Vector& xF0,
			    FiniteElement& element1, Mesh& mesh);

    static void computeFEuler(Vector& xF, Vector& xF0, Vector& xF1,
			      FiniteElement& element1, Mesh& mesh);

    static void computeFBEuler(Vector& xF, Vector& xB, Vector& xF0,
			       Vector& xF1,
			       FiniteElement& element1, Mesh& mesh);

    static void computeBEuler(Vector& xF, Vector& xB, 
			      FiniteElement& element1, Mesh& mesh);

    static void multF(real* F0, real *F1, real* F);
    
    static void multB(real* F, real* B);
    
    static void deform(Mesh& mesh, Function& u);
    
    // Data

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
    real lmbda;
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

//     ElasticityUpdated::LinearForm::TestElement element1;
//     ElasticityUpdatedSigma::LinearForm::TestElement element2;
//     ElasticityUpdatedSigma::LinearForm::FunctionElement_2 element3;

    FiniteElement* element1;
    FiniteElement* element2;
    FiniteElement* element3;

    Vector x1_0, x1_1, x2_0, x2_1, b, m, msigma, stepresidual;
    Vector xsigma0, xsigma1, xepsilon1, xsigmanorm, xjaumann1;
    Vector xtmp1, xtmp2, xsigmatmp1, xsigmatmp2;
    Vector fcontact;
    Matrix Dummy;

    Vector dotu_x1, dotu_x2, dotu_xsigma, dotu;

    VecScatter dotu_x1sc, dotu_x2sc, dotu_xsigmasc;
    IS dotu_x1is, dotu_x2is, dotu_xsigmais;
    


    int* dotu_x1_indices;
    int* dotu_x2_indices;
    int* dotu_xsigma_indices;



    Function v1;
    Function u0;
    Function u1;
    Function sigma0;
    Function sigma1;
    Function epsilon1;
    Function sigmanorm;

    // Forms

    // ElasticityUpdated::LinearForm Lv;
    // ElasticityUpdatedSigma::LinearForm Lsigma;

    LinearForm* Lv;
    LinearForm* Lsigma;
  };

  class ElasticityUpdatedODE : public ODE
  {
  public:
    ElasticityUpdatedODE(ElasticityUpdatedSolver& solver);
    void u0(uBlasVector& u);
    // Evaluate right-hand side (mono-adaptive version)

    // Fix to avoid error with some compilers due to only partially overridden
    // virtual functions
    using ODE::f; 
    virtual void f(const uBlasVector& u, real t, uBlasVector& y);
    virtual bool update(const uBlasVector& u, real t, bool end);

    ElasticityUpdatedSolver& solver;
  };

  // Boundary conditions
  // Temporary solution until BCs can be specified in forms

  class UtilBC1 : public BoundaryCondition
  {
  public:
    UtilBC1()
    {
    }
    
    void eval(BoundaryValue& value, const Point& p, unsigned int i)
    {
      if(p.x == 0.0)
	value = 0.0;
//       if(p.x < -0.8)
// 	value = 0.0;
    }
  };

  class UtilBC2 : public BoundaryCondition
  {
  public:
    UtilBC2()
    {
    }
    
    void eval(BoundaryValue& value, const Point& p, unsigned int i)
    {
//       if(p.x == 0.0)
// 	value = 0.0;
      if(p.x < -0.8)
	value = 0.0;
    }
  };


  // Resistance
  class Resistance : public Function
  {
  public:
    Resistance()
    {
    }
    
    real eval(const Point& p, unsigned int i)
    {
//       cout << "time: " << time() << endl;

      if(time() > 5.0 && time() < 5.4)
	return 1000;
      else
	return 0.0;
    }
  };


}

#endif

#endif

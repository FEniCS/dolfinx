//
// Created by Corrado Maurini, 2012.
//
// First added:  
// Last changed: 2012-12-03
//
// For an example of the implementation in C++ classes of a PETSc non-linear solve (snes), 
// see the libmesh interface : http://libmesh.sourceforge.net/doxygen/petsc__nonlinear__solver_8C_source.php
//

#ifndef _TAOLinearBoundSolver_H
#define _TAOLinearBoundSolver_H

#ifdef HAS_TAO

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/types.h>
#include <map>
#include <petscksp.h>
#include <petscpc.h>
#include <boost/shared_ptr.hpp>
#include <boost/scoped_ptr.hpp>
#include <tao.h>
#include <taosolver.h>
#include "PETScObject.h"

namespace dolfin
{


  /// Forward declarations
  class GenericMatrix;
  class GenericVector;
  class PETScMatrix;
  class PETScVector;

  /// This class provides bound constrained solver for a linear system defined by PETSc matrices and vectors:
  ///
  ///   Ax =  b, with xl =< x <= xu 
  ///
  /// It is a wrapper for the TAO bound constrained solver.
  ///
  /// The following parameters may be specified to control the solver.
  ///
  /// - "solver"
  ///
  /// This parameter controls the type of minimization algorithm used by TAO .
  /// Possible values for bound constrained solvers are
  ///	tao_tron	- Newton Trust Region method for bound constrained minimization
  ///	tao_gpcg	- Newton Trust Region method for quadratic bound constrained minimization
  ///   tao_blmvm	- Limited memory variable metric method for bound constrained minimization + tao_pounders - Model-based algorithm pounder extended for nonlinear least squares
  ///
  ///  - "monitor"
  /// This parameter controls the visualization of the convergence tests at each iteration of the solver. 
  /// Possible values are true or false
  /// 
  ///  - "report"
  /// This parameter controls the report of the final state of the solver at the end of the solving process. 
  //  Possible values are true or false
  /// 
  ///  - tolerances of the solver: "fatol", "frtol", "gatol", "grtol", "gttol"
  ///
  /// These parameters control the tolerances used by TAO.
  /// Possible values are positive double numbers. below their definition and default values 
  /// where f is the function to minimize and g its gradient
  ///
  /// f(X) - f(X*) (estimated)            <= fatol, default = 1e-10  
  /// |f(X) - f(X*)| (estimated) / |f(X)| <= frtol, default = 1e-10 
  /// ||g(X)||                            <= gatol, default = 1e-8 
  /// ||g(X)|| / |f(X)|                   <= grtol, default = 1e-8 
  /// ||g(X)|| / ||g(X0)||                <= gttol, default = 0   
  ///
  /// - "krylov_solver"
  /// This parameter set contains parameters to configure the PETSc Krylov solver used by Tao
  /// 
   class TAOLinearBoundSolver : public Variable, public PETScObject
  {
  public:
    /// Create TAO bound constrained solver 
    TAOLinearBoundSolver(std::string method="default");

    /// Destructor
    ~TAOLinearBoundSolver();

	// Set operators with GenericMatrix and GenericVector
	void set_operators(const boost::shared_ptr<const GenericMatrix> A,
                                      const boost::shared_ptr<const GenericVector> b);

	// Set operators with shared pointer to PETSc objects
	void set_operators(const boost::shared_ptr<const PETScMatrix> A,
                                      const boost::shared_ptr<const PETScVector> b); 
	     
    /// Solve linear system Ax = b with xl =< x <= xu 
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b, const GenericVector& xl, const GenericVector& xu);

    /// Solve linear system Ax = b with xl =< x <= xu 
    uint solve(const PETScMatrix& A, PETScVector& x, const PETScVector& b, const PETScVector& xl, const PETScVector& xu);
        
    // Set the solver type
	void set_solver(std::string solver);
	    	
	// Return PETSc KSP pointer
    boost::shared_ptr<TaoSolver> tao() const;
	
    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("tao_solver");

      p.add("method","tao_tron");
      p.add("monitor",false);
      p.add("report",true);
      p.add("fatol",1.0e-10);
      p.add("frtol",1.0e-10);
      p.add("gatol",1.0e-8);
      p.add("grtol",1.0e-8);
      p.add("gttol",0.);
      
      Parameters q("krylov_solver");
      q.add("method","default");
      q.add("nonzero_initial_guess",false);
      q.add("gmres_restart",30);
      q.add("monitor_convergence",false);
      q.add("relative_tolerance",1.0e-10);
      q.add("absolute_tolerance",1.0e-10);
      q.add("divergence_limit",10.0e+7);
      q.add("maximum_iterations",100);
      p.add(q);
      
      return p;
    }
    
    // Operator and vectors
    boost::shared_ptr<const PETScMatrix> A;
    boost::shared_ptr<const PETScVector> b;    
    
  private:
    /// Callback for changes in parameter values
    void read_parameters();

    // Set tolerance
    //void set_tolerances(double fatol, double frtol, double gatol, double grtol, double gttol);
   
    /// Tao solver pointer
    boost::shared_ptr<TaoSolver> _tao;
    
    // Available ksp solvers
    static const std::map<std::string, const KSPType> _ksp_methods;
    
    // Set options
    void set_ksp_options();
    
    // Initialize KSP solver
    void init(const std::string& method);

    // TAO solver pointer
    //TaoSolver tao;
    
};
}
//-----------------------------------------------------------------------------
//#undef __FUNCT__
//#define __FUNCT__ "TAOFormFunctionGradientQuadraticProblem"
/// Computes the value of the objective function and its gradient. 

PetscErrorCode TAOFormFunctionGradientQuadraticProblem(TaoSolver tao, Vec X, PetscReal *ener, Vec G, void *ptr);

//-----------------------------------------------------------------------------

//#undef __FUNCT__
//#define __FUNCT__ "TAOFormHessianQuadraticProblem"
/// Computes the hessian of the quadratic objective function 
/// Notice that the objective function in this problem is quadratic (therefore a constant hessian). 

PetscErrorCode TAOFormHessianQuadraticProblem(TaoSolver tao,Vec X,Mat *H, Mat *Hpre, MatStructure *flg, void *ptr);

//-----------------------------------------------------------------------------
//#undef __FUNCT__
//#define __FUNCT__ "TAOMonitor"
//  Monitor the state of the solution at each iteration. The output printed to the screen is:
//
//	iterate 	- the current iterate number (>=0)
//	f 	- the current function value
//	gnorm 	- the square of the gradient norm, duality gap, or other measure indicating distance from optimality.
//	cnorm 	- the infeasibility of the current solution with regard to the constraints.
//	xdiff 	- the step length or trust region radius of the most recent iterate. 
PetscErrorCode TAOMonitor(TaoSolver tao, void *ctx);

//-----------------------------------------------------------------------------
#endif

#endif

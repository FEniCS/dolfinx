//
// Created by Corrado Maurini, 2012.
//
// First added:  2012-12-03
// Last changed: 
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
#include "KrylovSolver.h"

namespace dolfin
{


  /// Forward declarations
  class GenericMatrix;
  class GenericVector;
  class PETScMatrix;
  class PETScVector;
  class PETScPreconditioner;
  class PETScKrylovSolver;
  class PETScKSPDeleter;

  /// This class provides bound constrained solver for a linear system defined by PETSc matrices and vectors:
  ///
  ///   Ax =  b, with xl =< x <= xu 
  ///
  /// It is a wrapper for the TAO bound constrained solver.
  ///
  /// Python example:
  /// 
  ///  ----------------------------------------------------------------------
  ///  Begin of python example
  ///  ----------------------------------------------------------------------
  ///  # Assemble the linear system
  ///     A, b = assemble_system(a, L, bc)
  ///  # Define the constraints
  ///     constraint_u = Constant(1.)
  ///     constraint_l = Constant(0.)
  ///     u_min = interpolate(constraint_l, V)
  ///     u_max = interpolate(constraint_u, V)
  ///  # Define the function to store the solution 
  ///     usol=Function(V)
  ///  # Create the TAOLinearBoundSolver
  ///    solver=TAOLinearBoundSolver("tao_gpcg","gmres")
  ///  # Set some parameters
  ///    solver.parameters["monitor_convergence"]=True
  ///    solver.parameters["report"]=True
  ///  # Solve the problem
  ///    solver.solve(A, usol.vector(), b , u_min.vector(), u_max.vector())
  ///  ----------------------------------------------------------------------
  ///  End of python example
  ///  ----------------------------------------------------------------------
  ///
  /// To get a list of available parameters: 
  ///
  /// info(solver.parameters,True)
  ///
   class TAOLinearBoundSolver : public Variable, public PETScObject
  {
  public:
    /// Create TAO bound constrained solver 
    TAOLinearBoundSolver(const std::string method   = "default" ,
                         const std::string ksp_type = "default" , 
                         const std::string pc_type  = "default" );

    /// Destructor
    ~TAOLinearBoundSolver();
	     
    /// Solve linear system Ax = b with xl =< x <= xu 
    uint solve(const GenericMatrix& A, GenericVector& x, const GenericVector& b, const GenericVector& xl, const GenericVector& xu);

    /// Solve linear system Ax = b with xl =< x <= xu 
    uint solve(const PETScMatrix& A, PETScVector& x, const PETScVector& b, const PETScVector& xl, const PETScVector& xu);
        
    // Set the TAO solver type
	void set_solver(const std::string&);
	
   /// Set PETSC Krylov Solver (ksp) used by TAO
    void set_ksp( const std::string ksp_type = "default");     	
	    	
	// Return TAO solver pointer
    boost::shared_ptr<TaoSolver> tao() const;
    
    /// Return a list of available Tao solver methods
    static std::vector<std::pair<std::string, std::string> > methods();
    
    /// Return a list of available krylov solvers
    static std::vector<std::pair<std::string, std::string> > krylov_solvers();

    /// Return a list of available preconditioners
    static std::vector<std::pair<std::string, std::string> > preconditioners();
    
    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("tao_solver");

      p.add("monitor_convergence"    , false   );
      p.add("report"                 , false   );
      p.add("function_absolute_tol"  , 1.0e-10 );
      p.add("function_relative_tol"  , 1.0e-10 );
      p.add("gradient_absolute_tol"  , 1.0e-8  );
      p.add("gradient_relative_tol"  , 1.0e-8  );
      p.add("gradient_t_tol"         , 0.      );
      p.add("error_on_nonconvergence", true    );
      p.add("maximum_iterations"     , 100     );
      
      Parameters ksp("krylov_solver");
      ksp = KrylovSolver::default_parameters();
      p.add(ksp);
      
      return p;
    }
    
    // Return Matrix shared pointer
    boost::shared_ptr<const PETScMatrix> get_matrix() const;

    // Return load vector shared pointer
    boost::shared_ptr<const PETScVector> get_vector() const;


  private:
    
    // Set operators with GenericMatrix and GenericVector
	void set_operators(const boost::shared_ptr<const GenericMatrix> A,
                                      const boost::shared_ptr<const GenericVector> b);

	// Set operators with shared pointer to PETSc objects
	void set_operators(const boost::shared_ptr<const PETScMatrix> A,
                                      const boost::shared_ptr<const PETScVector> b); 
        
    // Callback for changes in parameter values
    void read_parameters();

    // Available ksp solvers 
    static const std::map<std::string, const KSPType> _ksp_methods;

    // Available tao solvers descriptions
    static const std::vector<std::pair<std::string, std::string> > _methods_descr;
    
    // Set options
    void set_ksp_options();
    
    // Initialize TAO solver
    void init(const std::string& method); 
    
    // Petsc preconditioner
    boost::shared_ptr<PETScPreconditioner> preconditioner;

    // Tao solver pointer
    boost::shared_ptr<TaoSolver> _tao;
    
    // Operator (the matrix) and the vector
    boost::shared_ptr<const PETScMatrix> A;
    boost::shared_ptr<const PETScVector> b;   
    
    bool preconditioner_set;   
        
    /// Computes the value of the objective function and its gradient. 
    static PetscErrorCode __TAOFormFunctionGradientQuadraticProblem(TaoSolver tao, Vec X, PetscReal *ener, Vec G, void *ptr);
    
    /// Computes the hessian of the quadratic objective function 
    static PetscErrorCode __TAOFormHessianQuadraticProblem(TaoSolver tao,Vec X,Mat *H, Mat *Hpre, MatStructure *flg, void *ptr);
    
    //-----------------------------------------------------------------------------
    //  Monitor the state of the solution at each iteration. The output printed to the screen is:
    //
    //	iterate 	- the current iterate number (>=0)
    //	f 	- the current function value
    //	gnorm 	- the square of the gradient norm, duality gap, or other measure indicating distance from optimality.
    //	cnorm 	- the infeasibility of the current solution with regard to the constraints.
    //	xdiff 	- the step length or trust region radius of the most recent iterate. 
    //-----------------------------------------------------------------------------
    static PetscErrorCode __TAOMonitor(TaoSolver tao, void *ctx);
 
  };

}
#endif

#endif

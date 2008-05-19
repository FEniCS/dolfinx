// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Last changed: 2008-05-16

#ifdef HAS_TRILINOS

#include <dolfin/log/dolfin_log.h>
#include "EpetraKrylovSolver.h"
#include "EpetraMatrix.h"
#include "EpetraVector.h"

#include "Epetra_FEVector.h"
#include "Epetra_FECrsMatrix.h"
#include "Epetra_LinearProblem.h"

//#include "ml_config.h"
#include "ml_include.h"
#include "ml_MultiLevelOperator.h"
#include "ml_epetra_utils.h"

#include "AztecOO.h"




using namespace dolfin; 
//-----------------------------------------------------------------------------
EpetraKrylovSolver::EpetraKrylovSolver(SolverType method_, PreconditionerType pc_) : 
  method(method_), pc_type(pc_), prec(0) 
{ 
}
//-----------------------------------------------------------------------------
EpetraKrylovSolver::EpetraKrylovSolver(SolverType method_, EpetraPreconditioner& prec_) : 
  method(method_), pc_type(default_pc), prec(&prec_) 
{ 
}
//-----------------------------------------------------------------------------
EpetraKrylovSolver::~EpetraKrylovSolver() {}
//-----------------------------------------------------------------------------
dolfin::uint EpetraKrylovSolver::solve(const EpetraMatrix& A, EpetraVector& x, const EpetraVector& b) {
//FIXME need the ifdef AztecOO 

// create linear system 
  Epetra_LinearProblem linear_system(&(A.mat()),&(x.vec()),&(b.vec()));
  // create AztecOO instance
  AztecOO linear_solver(linear_system);

  if ( method == cg) { linear_solver.SetAztecOption( AZ_solver, AZ_cg); }
  else if ( method == gmres) {linear_solver.SetAztecOption( AZ_solver, AZ_gmres); }
  else if ( method == bicgstab) {linear_solver.SetAztecOption( AZ_solver, AZ_bicgstab); }
  else if ( method == lu ) { error("EpetraKrylovSolver::solve LU not supported"); } 

  if ( pc_type == jacobi) { linear_solver.SetAztecOption( AZ_precond, AZ_Jacobi); }
  //FIXME GS or SSOR not a PreconditionerType not in 
  else if ( pc_type == sor) {linear_solver.SetAztecOption( AZ_precond, AZ_sym_GS); }
  else if ( pc_type == ilu) {linear_solver.SetAztecOption( AZ_precond, AZ_ilu); }
  else if ( pc_type == icc) {linear_solver.SetAztecOption( AZ_precond, AZ_icc); }

  if (pc_type == amg) 
  {  
#ifdef HAVE_ML_AZTECOO

    //FIXME ifdef ML 
    //FIXME if amg 
    // Code from trilinos-8.0.3/packages/didasko/examples/ml/ex1.cpp 

    // Create and set an ML multilevel preconditioner
    ML *ml_handle;

    // Maximum number of levels
    int N_levels = 10;

    // output level
    ML_Set_PrintLevel(0);

    ML_Create(&ml_handle,N_levels);

    // wrap Epetra Matrix into ML matrix (data is NOT copied)
    EpetraMatrix2MLMatrix(ml_handle, 0, &(A.mat()));

    // create a ML_Aggregate object to store the aggregates
    ML_Aggregate *agg_object;
    ML_Aggregate_Create(&agg_object);

    // specify max coarse size 
    ML_Aggregate_Set_MaxCoarseSize(agg_object,1);

    // generate the hierady
    N_levels = ML_Gen_MGHierarchy_UsingAggregation(ml_handle, 0, ML_INCREASING, agg_object);

    // Set a symmetric Gauss-Seidel smoother for the MG method 
    ML_Gen_Smoother_SymGaussSeidel(ml_handle, ML_ALL_LEVELS, ML_BOTH, 1, ML_DEFAULT);

    // generate solver
    ML_Gen_Solver(ml_handle, ML_MGV, 0, N_levels-1);

    // wrap ML_Operator into Epetra_Operator
    ML_Epetra::MultiLevelOperator  MLop(ml_handle,A.mat().Comm(),A.mat().DomainMap(),A.mat().RangeMap());

    // set this operator as preconditioner for AztecOO
    linear_solver.SetPrecOperator(&MLop);

#else 
    error("EpetraKrylovSolver::solve not compiled with ML support."); 
#endif 
  }   
  linear_solver.Iterate(1000,1E-9);
  return linear_solver.NumIters(); 
}
//-----------------------------------------------------------------------------
void EpetraKrylovSolver::disp() const {
  error("EpetraKrylovSolver::disp not implemented"); 
}
//-----------------------------------------------------------------------------
#endif 



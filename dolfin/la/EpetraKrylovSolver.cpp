// Copyright (C) 2008 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2009.
//
// Last changed: 2009-09-08

#ifdef HAS_TRILINOS

#include <Epetra_ConfigDefs.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_FECrsMatrix.h>
#include <Epetra_FEVector.h>
#include <Epetra_LinearProblem.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_Vector.h>
#include <Epetra_Map.h>
#include <AztecOO.h>
#include <ml_include.h>
#include <Epetra_LinearProblem.h>
#include <ml_MultiLevelOperator.h>
#include <ml_epetra_utils.h>

#include <boost/assign/list_of.hpp>
#include <dolfin/log/dolfin_log.h>
#include "GenericMatrix.h"
#include "GenericVector.h"
#include "EpetraKrylovSolver.h"
#include "EpetraMatrix.h"
#include "EpetraVector.h"
#include "EpetraUserPreconditioner.h"
#include "KrylovSolver.h"

using namespace dolfin;

// Available solvers
const std::map<std::string, int> EpetraKrylovSolver::methods
  = boost::assign::map_list_of("default",  AZ_gmres)
                              ("cg",       AZ_cg)
                              ("gmres",    AZ_gmres)
                              ("bicgstab", AZ_bicgstab);

// Available preconditioners
const std::map<std::string, int> EpetraKrylovSolver::pc_methods
  = boost::assign::map_list_of("default", AZ_ilu)
                              ("ilu",     AZ_ilu)
                              ("jacobi",  AZ_Jacobi)
                              ("sor",     AZ_sym_GS)
                              ("icc",     AZ_icc)
                              ("amg_ml",  -1);

//-----------------------------------------------------------------------------
Parameters EpetraKrylovSolver::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters());
  p.rename("epetra_krylov_solver");
  return p;
}
//-----------------------------------------------------------------------------
EpetraKrylovSolver::EpetraKrylovSolver(std::string method, std::string pc_type)
                    : method(method), pc_type(pc_type), prec(0)
{
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
EpetraKrylovSolver::EpetraKrylovSolver(std::string method,
                                       EpetraUserPreconditioner& prec)
                                     : method(method), pc_type("default"),
                                       prec(&prec)
{
  error("Initialisation of EpetraKrylovSolver with a EpetraUserPreconditioner needs to be implemented.");
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
EpetraKrylovSolver::~EpetraKrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraKrylovSolver::solve(const GenericMatrix& A, GenericVector& x,
                                       const GenericVector& b)
{
  return solve(A.down_cast<EpetraMatrix>(), x.down_cast<EpetraVector>(),
               b.down_cast<EpetraVector>());
}
//-----------------------------------------------------------------------------
dolfin::uint EpetraKrylovSolver::solve(const EpetraMatrix& A, EpetraVector& x,
                                       const EpetraVector& b)
{
  // FIXME: This function needs to be cleaned up

  //cout << "!!!Inside solve " << endl;

  // Check that requsted solver is supported
  if (methods.count(method) == 0 )
    error("Requested EpetraKrylovSolver method '%s' in unknown", method.c_str());

  // Check that requsted preconditioner is supported
  if (pc_methods.count(pc_type) == 0 )
    error("Requested EpetraKrylovSolver preconditioner '%s' in unknown", pc_type.c_str());

  //FIXME: We need to test for AztecOO during configuration

  // Check dimensions
  uint M = A.size(0);
  uint N = A.size(1);
  if (N != b.size())
    error("Non-matching dimensions for linear system.");

  // Write a message
  if (parameters["report"])
    info("Solving linear system of size %d x %d (Krylov solver).", M, N);

  // Reinitialize solution vector if necessary
  if (x.size() != M)
  {
    x.resize(M);
    x.zero();
  }

  //cout << "Stage A " << endl;

  // FIXME: permit initial guess

  //cout << "Stage A(i) " << endl;

  // Create linear problem
  Epetra_LinearProblem linear_problem(A.mat().get(), x.vec().get(),
                                      b.vec().get());

  //cout << "Stage A(ii) " << endl;

  // Create linear solver
  AztecOO linear_solver(linear_problem);

  //cout << "Stage B " << endl;

  // Set solver type
  linear_solver.SetAztecOption(AZ_solver, methods.find(method)->second);

  //cout << "Stage C " << endl;

  // Set output level
  if(parameters["monitor_convergence"])
   linear_solver.SetAztecOption(AZ_output, 1);
  else
    linear_solver.SetAztecOption(AZ_output, AZ_none);

  //cout << "Stage D " << endl;

  // Set preconditioner
  if (pc_type == "default" || pc_type == "ilu")
  {
    linear_solver.SetAztecOption(AZ_precond, AZ_dom_decomp);
    linear_solver.SetAztecOption(AZ_subdomain_solve, pc_methods.find(pc_type)->second);
  }
  else if(pc_type != "amg_ml")
    linear_solver.SetAztecOption(AZ_precond, pc_methods.find(pc_type)->second);
  else if (pc_type == "amg_ml")
  {
    //cout << "Stage E " << endl;

    // FIXME: Move configuration of ML to another function
    //error("The EpetraKrylovSolver interface for the ML preconditioner needs to be fixed.");

    #ifdef HAVE_ML_AZTECOO
    // Code from trilinos-8.0.3/packages/didasko/examples/ml/ex1.cpp

    // Create and set an ML multilevel preconditioner
    ML *ml_handle;

    // Maximum number of levels
    int N_levels = 10;

    // output level
    ML_Set_PrintLevel(0);

    ML_Create(&ml_handle,N_levels);

    // Wrap Epetra Matrix into ML matrix (data is NOT copied)
    EpetraMatrix2MLMatrix(ml_handle, 0, A.mat().get());

    // Create a ML_Aggregate object to store the aggregates
    ML_Aggregate* agg_object;
    ML_Aggregate_Create(&agg_object);

    // Specify max coarse size
    ML_Aggregate_Set_MaxCoarseSize(agg_object, 1);

    // Generate the hierady
    N_levels = ML_Gen_MGHierarchy_UsingAggregation(ml_handle, 0, ML_INCREASING, agg_object);

    // Set a symmetric Gauss-Seidel smoother for the MG method
    ML_Gen_Smoother_SymGaussSeidel(ml_handle, ML_ALL_LEVELS, ML_BOTH, 1, ML_DEFAULT);

    // Generate solver
    ML_Gen_Solver(ml_handle, ML_MGV, 0, N_levels-1);

    // Wrap ML_Operator into Epetra_Operator
    ML_Epetra::MultiLevelOperator mLop(ml_handle, (*A.mat()).Comm(), (*A.mat()).DomainMap(), (*A.mat()).RangeMap());

    // Set this operator as preconditioner for AztecOO
    linear_solver.SetPrecOperator(&mLop);

    #else
    error("Epetra has not been compiled with ML support.");
    #endif
  }

  // Start solve
  cout << "Start solve " << endl;
  linear_solver.Iterate(parameters["maximum_iterations"], parameters["relative_tolerance"]);
  cout << "End solve " << endl;

  info("AztecOO Krylov solver (%s, %s) converged in %d iterations.",
          method.c_str(), pc_type.c_str(), linear_solver.NumIters());

  return linear_solver.NumIters();
}
//-----------------------------------------------------------------------------
std::string EpetraKrylovSolver::str(bool verbose) const
{
  dolfin_not_implemented();
  return std::string();
}
//-----------------------------------------------------------------------------

#endif

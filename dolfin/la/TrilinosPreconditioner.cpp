// Copyright (C) 2010 Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-25
// Last changed:

#ifdef HAS_TRILINOS

#include <boost/assign/list_of.hpp>
#include <AztecOO.h>
#include <Ifpack.h>
#include <Epetra_CombineMode.h>

#include <ml_include.h>
#include <ml_MultiLevelOperator.h>
#include <ml_epetra_utils.h>

#include <dolfin/log/dolfin_log.h>
#include "EpetraKrylovSolver.h"
#include "KrylovSolver.h"
#include "TrilinosPreconditioner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// Available preconditioners
const std::map<std::string, int> TrilinosPreconditioner::methods
  = boost::assign::map_list_of("default", AZ_ilu)
                              ("ilu",     AZ_ilu)
                              ("jacobi",  AZ_Jacobi)
                              ("none",    AZ_none)
                              ("sor",     AZ_sym_GS)
                              ("icc",     AZ_icc)
                              ("amg_ml",  -1);
//-----------------------------------------------------------------------------
Parameters TrilinosPreconditioner::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters());
  p.rename("trilinos_preconditioner");
  p.add("ilu_fill_level", 0);
  p.add("schwarz_overlap", 1);
  return p;
}
//-----------------------------------------------------------------------------
TrilinosPreconditioner::TrilinosPreconditioner(std::string type) : type(type)
{
  // Set parameter values
  parameters = default_parameters();

  // Check that the requested method is known
  if (methods.count(type) == 0)
    error("Requested PETSc proconditioner '%s' is unknown,", type.c_str());
}
//-----------------------------------------------------------------------------
TrilinosPreconditioner::~TrilinosPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void TrilinosPreconditioner::set(EpetraKrylovSolver& solver)
{
  assert(solver.aztecoo());

  // Get underlying solver object
  AztecOO& _solver = *(solver.aztecoo());

  // Set preconditioner
  if (type == "default" || type == "ilu")
  {
    /*
    const int ilu_fill_level = parameters["ilu_fill_level"];
    const int overlap        = parameters["schwarz_overlap"];
    Teuchos::ParameterList list;
    list.set("fact: level-of-fill", ilu_fill_level);
    list.set("schwarz: combine mode", "Zero");

    Epetra_RowMatrix* A = _solver.GetUserMatrix();

    Ifpack ifpack_factory;
    string type = "ILU";
    ifpack_preconditioner.reset( ifpack_factory.Create(type, A, overlap) );
    assert(ifpack_preconditioner != 0);

    ifpack_preconditioner->SetParameters(list);
    ifpack_preconditioner->Initialize();
    ifpack_preconditioner->Compute();
    _solver.SetPrecOperator(ifpack_preconditioner.get());
    std::cout << *ifpack_preconditioner;
    */

    _solver.SetAztecOption(AZ_precond, AZ_dom_decomp);
    _solver.SetAztecOption(AZ_subdomain_solve, methods.find(type)->second);
    _solver.SetAztecOption(AZ_graph_fill, parameters["ilu_fill_level"]);
    //_solver.CheckInput();
  }
  else if (type == "amg_ml")
    set_ml(_solver);
  else
    _solver.SetAztecOption(AZ_precond, methods.find(type)->second);
}
//-----------------------------------------------------------------------------
std::string TrilinosPreconditioner::name() const
{
  return type;
}
//-----------------------------------------------------------------------------
std::string TrilinosPreconditioner::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
    warning("Verbose output for TrilinosPreconditioner not implemented.");
  else
    s << "<TrilinosPreconditioner>";

  return s.str();
}
//-----------------------------------------------------------------------------
void TrilinosPreconditioner::set_ml(AztecOO& solver) const
{
  warning("The TrilinosPreconditioner interface for the ML preconditioner needs to be fixed.");

  // FIXME: Check that GetUserMatrix returns what we want
  Epetra_RowMatrix* A = solver.GetUserMatrix();
  assert(A);

  #ifdef HAVE_ML_AZTECOO
  // Code from trilinos-8.0.3/packages/didasko/examples/ml/ex1.cpp

  // Create and set an ML multilevel preconditioner
  ML *ml_handle;

  // Maximum number of levels
  int N_levels = 10;

  // output level
  ML_Set_PrintLevel(0);

  ML_Create(&ml_handle, N_levels);

  // Wrap Epetra Matrix into ML matrix (data is NOT copied)
  EpetraMatrix2MLMatrix(ml_handle, 0, A);

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
  ML_Gen_Solver(ml_handle, ML_MGV, 0, N_levels - 1);

  // Wrap ML_Operator into Epetra_Operator
  ML_Epetra::MultiLevelOperator mLop(ml_handle, A->Comm(), A->OperatorDomainMap(), A->OperatorRangeMap());

  // Set this operator as preconditioner for AztecOO
  solver.SetPrecOperator(&mLop);

  #else
  error("Epetra has not been compiled with ML support.");
  #endif
}
//-----------------------------------------------------------------------------

#endif

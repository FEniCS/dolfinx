// Copyright (C) 2010 Garth N. Wells
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-02-25
// Last changed: 2010-03-03

#ifdef HAS_TRILINOS

#include <boost/assign/list_of.hpp>
#include <AztecOO.h>
#include <Ifpack.h>
#include <Epetra_CombineMode.h>
#include <ml_include.h>
#include <ml_epetra_utils.h>
#include "ml_MultiLevelPreconditioner.h"
#include "Teuchos_ParameterList.hpp"

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
  p.add("schwarz_mode", "Zero");   // Options are Zero, Insert, Add, Average, AbsMax
  p.add("reordering_type", "rcm"); // Options are rcm, metis, amd
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
  if (type == "default" || type == "ilu" || type == "icc")
  {
    // Get/set some parameters
    const int ilu_fill_level       = parameters["ilu_fill_level"];
    const int overlap              = parameters["schwarz_overlap"];
    const std::string reordering   = parameters["reordering_type"];
    const std::string schwarz_mode = parameters["schwarz_mode"];
    Teuchos::ParameterList list;
    list.set("fact: level-of-fill",      ilu_fill_level);
    list.set("schwarz: combine mode",    schwarz_mode);
    list.set("schwarz: reordering type", reordering);

    Epetra_RowMatrix* A = _solver.GetUserMatrix();

    // Create preconditioner
    std::string type;
    if (type == "icc")
      type = "IC";
    else
      type = "ILU";
    Ifpack ifpack_factory;
    ifpack_preconditioner.reset( ifpack_factory.Create(type, A, overlap) );
    assert(ifpack_preconditioner != 0);

    // Set up preconditioner
    ifpack_preconditioner->SetParameters(list);
    ifpack_preconditioner->Initialize();
    ifpack_preconditioner->Compute();
    _solver.SetPrecOperator(ifpack_preconditioner.get());

    //std::cout << *ifpack_preconditioner;
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
void TrilinosPreconditioner::set_ml(AztecOO& solver)
{
  warning("The TrilinosPreconditioner interface for the ML preconditioner is experimental.");

  // FIXME: Check that GetUserMatrix returns what we want
  Epetra_RowMatrix* A = solver.GetUserMatrix();
  assert(A);

  Teuchos::ParameterList mlist;

  //ML_Epetra::SetDefaults("SA", mlist);
  ML_Epetra::SetDefaults("DD", mlist);
  //mlist.set("max levels", 10);
  //mlist.set("increasing or decreasing", "decreasing");
  mlist.set("aggregation: type", "ParMETIS");
  //mlist.set("coarse: max size", 30);

  //mlist.set("aggregation: nodes per aggregate", 4);
  //mlist.set("coarse: type","Amesos-KLU");
  mlist.set("coarse: type", "Amesos-UMFPACK");

  //mlist.set("max levels", 6);
  mlist.set("ML output", 10);

  // Create preconditioner
  ml_preconditioner.reset(new ML_Epetra::MultiLevelPreconditioner(*A, mlist, true));

  // Set this operator as preconditioner for AztecOO
  solver.SetPrecOperator(ml_preconditioner.get());
}
//-----------------------------------------------------------------------------

#endif

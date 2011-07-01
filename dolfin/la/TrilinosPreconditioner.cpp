// Copyright (C) 2010 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2010-02-25
// Last changed: 2011-06-30

#ifdef HAS_TRILINOS

#include <dolfin/common/MPI.h>
#include <boost/assign/list_of.hpp>

#include <AztecOO.h>
#include <Epetra_CombineMode.h>
#include <Epetra_FECrsMatrix.h>
#include <Epetra_RowMatrix.h>
#include <Ifpack.h>
#include <ml_include.h>
#include <ml_epetra_utils.h>
#include <ml_MultiLevelPreconditioner.h>
#include <Teuchos_ParameterList.hpp>

#include <dolfin/log/dolfin_log.h>
#include "EpetraKrylovSolver.h"
#include "EpetraMatrix.h"
#include "KrylovSolver.h"
#include "TrilinosPreconditioner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// Available preconditioners
const std::map<std::string, int> TrilinosPreconditioner::methods
  = boost::assign::map_list_of("default",   AZ_ilu)
                              ("ilu",       AZ_ilu)
                              ("jacobi",    AZ_Jacobi)
                              ("none",      AZ_none)
                              ("sor",       AZ_sym_GS)
                              ("icc",       AZ_icc)
                              ("amg",       -1)
                              ("hypre_amg", -1)
                              ("ml_amg",    -1);
//-----------------------------------------------------------------------------
Parameters TrilinosPreconditioner::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters()("preconditioner"));
  p.rename("trilinos_preconditioner");

  // Add some extra Trilinos-specific Schwarz options
  //    Options are Zero, Insert, Add, Average, AbsMax
  p("schwarz").add("mode", "Zero");
  //    Options are rcm, metis, amd
  p("schwarz").add("reordering_type", "rcm");

  // ML options
  Parameters p_ml("ml");
  p_ml.add("output_level", 0);
  p_ml.add("max_levels", 10);

  // Add nested parameters sets
  p.add(p_ml);

  return p;
}
//-----------------------------------------------------------------------------
TrilinosPreconditioner::TrilinosPreconditioner(std::string type) : type(type)
{
  // Set parameter values
  parameters = default_parameters();

  // Check that the requested method is known
  if (methods.count(type) == 0)
    error("Requested Trilinos proconditioner '%s' is unknown,", type.c_str());
}
//-----------------------------------------------------------------------------
TrilinosPreconditioner::~TrilinosPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void TrilinosPreconditioner::set(EpetraKrylovSolver& solver,
                                 const EpetraMatrix& P)
{
  assert(solver.aztecoo());

  // Pointer to preconditioner matrix
  Epetra_RowMatrix* _P = P.mat().get();

  // Get underlying solver object
  AztecOO& _solver = *(solver.aztecoo());

  // Set preconditioner
  if (type == "default" || type == "ilu" || type == "icc")
  {
    // Get/set some parameters
    const int ilu_fill_level       = parameters("ilu")["fill_level"];
    const int overlap              = parameters("schwarz")["overlap"];
    const std::string reordering   = parameters("schwarz")["reordering_type"];
    const std::string schwarz_mode = parameters("schwarz")["mode"];
    Teuchos::ParameterList list;
    list.set("fact: level-of-fill",      ilu_fill_level);
    list.set("schwarz: combine mode",    schwarz_mode);
    list.set("schwarz: reordering type", reordering);

    // Create preconditioner
    std::string type;
    if (type == "icc")
      type = "IC";
    else
      type = "ILU";
    Ifpack ifpack_factory;
    ifpack_preconditioner.reset(ifpack_factory.Create(type, _P, overlap));
    assert(ifpack_preconditioner != 0);

    // Set up preconditioner
    ifpack_preconditioner->SetParameters(list);
    ifpack_preconditioner->Initialize();
    ifpack_preconditioner->Compute();
    _solver.SetPrecOperator(ifpack_preconditioner.get());
  }
  else if (type == "hypre_amg")
  {
    info("Hypre AMG not available for Trilinos. Using ML instead.");
    set_ml(_solver, *_P);
  }
  else if (type == "ml_amg" || type == "amg")
    set_ml(_solver, *_P);
  else
  {
    _solver.SetAztecOption(AZ_precond, methods.find(type)->second);
    _solver.SetPrecMatrix(_P);
  }
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
void TrilinosPreconditioner::set_ml(AztecOO& solver, const Epetra_RowMatrix& P)
{
  warning("The TrilinosPreconditioner interface for the ML preconditioner is experimental.");

  Teuchos::ParameterList mlist;

  //ML_Epetra::SetDefaults("SA", mlist);
  //ML_Epetra::SetDefaults("DD", mlist);
  //mlist.set("increasing or decreasing", "decreasing");
  //mlist.set("aggregation: type", "ParMETIS");
  //mlist.set("coarse: max size", 30);

  //mlist.set("aggregation: nodes per aggregate", 4);
  //mlist.set("coarse: type","Amesos-KLU");
  //mlist.set("coarse: type", "Amesos-UMFPACK");

  // Set maximum numer of level
  //const int max_levels = parameters("ml")["max_levels"];
  //mlist.set("max levels", max_levels);

  // Set output level
  const int output_level = parameters("ml")["output_level"];
  mlist.set("ML output", output_level);

  // Create preconditioner (assumes the A has been created)
  ml_preconditioner.reset(new ML_Epetra::MultiLevelPreconditioner(P, mlist, true));

  // Set this operator as preconditioner for AztecOO
  solver.SetPrecOperator(ml_preconditioner.get());
}
//-----------------------------------------------------------------------------

#endif

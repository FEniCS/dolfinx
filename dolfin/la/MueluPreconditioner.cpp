// Copyright (C) 2015 Chris Richardson
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

#ifdef HAS_TRILINOS

#include <MueLu_CreateTpetraPreconditioner.hpp>
#include "BelosKrylovSolver.h"
#include "KrylovSolver.h"
#include "MueluPreconditioner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MueluPreconditioner::MueluPreconditioner()
{
  // Set parameters
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
MueluPreconditioner::~MueluPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MueluPreconditioner::init(std::shared_ptr<const TpetraMatrix> P)
{
  Teuchos::ParameterList paramList;
  paramList.set("verbosity", "extreme");

  paramList.set("max levels", 10);
  paramList.set("coarse: max size", 10);
  paramList.set("coarse: type", "DIRECT");
  paramList.set("multigrid algorithm", "unsmoothed");

  Teuchos::ParameterList pre_paramList;
  pre_paramList.set("relaxation: type", "Symmetric Gauss-Seidel");
  pre_paramList.set("relaxation: sweeps", 3);
  pre_paramList.set("relaxation: damping factor", 0.6);
  paramList.set("smoother: pre type", "RELAXATION");
  paramList.set("smoother: pre params", pre_paramList);

  Teuchos::ParameterList post_paramList;
  post_paramList.set("relaxation: type", "Gauss-Seidel");
  post_paramList.set("relaxation: sweeps", 1);
  post_paramList.set("relaxation: damping factor", 0.9);
  paramList.set("smoother: post type", "RELAXATION");
  paramList.set("smoother: post params", post_paramList);

  // paramList.set("aggregation: type", "uncoupled");
  // paramList.set("aggregation: min agg size", 3);
  // paramList.set("aggregation: max agg size", 9);

  // FIXME: why does it need to be non-const when Ifpack2 uses const?
  std::shared_ptr<TpetraMatrix> P_non_const
    = std::const_pointer_cast<TpetraMatrix>(P);

  _prec = MueLu::CreateTpetraPreconditioner(
                Teuchos::rcp_dynamic_cast<op_type>(P_non_const->mat()),
                paramList);
}
//-----------------------------------------------------------------------------
void MueluPreconditioner::set(BelosKrylovSolver& solver)
{
  solver._problem->setLeftPrec(_prec);
}
//-----------------------------------------------------------------------------
std::string MueluPreconditioner::str(bool verbose) const
{
  std::stringstream s;
  s << "<MueluPreconditioner>";
  if (verbose)
    s << _prec->description() << std::endl;

  return s.str();
}
//-----------------------------------------------------------------------------
Parameters MueluPreconditioner::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters()("preconditioner"));
  p.rename("muelu_preconditioner");

  Teuchos::RCP<const Teuchos::ParameterList> pList = MueLu::MasterList::List();
  pList->print();

  return p;
}
//-----------------------------------------------------------------------------
#endif

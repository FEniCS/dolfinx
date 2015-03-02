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

#include "KrylovSolver.h"
#include "BelosKrylovSolver.h"
#include "MueluPreconditioner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MueluPreconditioner::MueluPreconditioner()
{
}
//-----------------------------------------------------------------------------
MueluPreconditioner::~MueluPreconditioner()
{
}
//-----------------------------------------------------------------------------
void MueluPreconditioner::init(std::shared_ptr<const TpetraMatrix> P)
{
  Teuchos::ParameterList paramList;
  paramList.set("verbosity", "extreme");

  paramList.set("max levels", 3);
  paramList.set("coarse: max size", 10);
  paramList.set("coarse: type", "RILUK");

  paramList.set("multigrid algorithm", "sa");

  paramList.set("smoother: type", "RELAXATION");
  Teuchos::ParameterList sparamList;
   sparamList.set("relaxation: type", "Jacobi");
   sparamList.set("relaxation: sweeps", 1);
   sparamList.set("relaxation: damping factor", 0.9);
  paramList.set("smoother: params", sparamList);

  paramList.set("aggregation: type", "uncoupled");
  paramList.set("aggregation: min agg size", 3);
  paramList.set("aggregation: max agg size", 9);

  _prec = MueLu::CreateTpetraPreconditioner(std::const_pointer_cast<TpetraMatrix>(P)->mat(), paramList);
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

  return p;
}
//-----------------------------------------------------------------------------
#endif

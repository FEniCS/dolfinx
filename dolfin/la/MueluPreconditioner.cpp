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
#include "TrilinosParameters.h"

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
  // Generate Trilinos parameters from dolfin parameters
  Teuchos::RCP<Teuchos::ParameterList> paramList(new Teuchos::ParameterList);
  TrilinosParameters::insert_parameters(parameters, paramList);

  // FIXME: why does it need to be non-const when Ifpack2 uses const?
  std::shared_ptr<TpetraMatrix> P_non_const
    = std::const_pointer_cast<TpetraMatrix>(P);

  _prec = MueLu::CreateTpetraPreconditioner(
                Teuchos::rcp_dynamic_cast<op_type>(P_non_const->mat()),
                *paramList);
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

  // Print off all the possible parameters
  // FIXME: pipe this to stringstream and output when verbose is set
  // Teuchos::RCP<const Teuchos::ParameterList> pList = MueLu::MasterList::List();
  // pList->print();

  return s.str();
}
//-----------------------------------------------------------------------------
Parameters MueluPreconditioner::default_parameters()
{
  Parameters p("muelu_preconditioner");
  p.rename("muelu_preconditioner");
  p.add("verbosity", "low");

  return p;
}
//-----------------------------------------------------------------------------
#endif

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

#include "KrylovSolver.h"
#include "BelosKrylovSolver.h"
#include "Ifpack2Preconditioner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::map<std::string, std::string>
Ifpack2Preconditioner::preconditioners()
{
  std::map<std::string, std::string> precs_available
    =   { {"none", "none"},
          {"default",    "default preconditioner"},
          {"diagonal",   "Diagonal"},
          {"relaxation", "Jacobi and Gauss-Seidel type relaxation"},
          {"chebyshev",  "Chebyshev Polynomial preconditioner"},
          {"riluk",      "Relaxed ILU with level k fill"},
          {"krylov",     "CG/GMRES with zero initial guess"}};
  return precs_available;
}
//-----------------------------------------------------------------------------
Ifpack2Preconditioner::Ifpack2Preconditioner(std::string preconditioner)
  : _name(preconditioner)
{
  // Check that the requested method is known
  const std::map<std::string, std::string> _preconditioners = preconditioners();
  if (_preconditioners.find(preconditioner) == _preconditioners.end())
  {
    dolfin_error("Ifpack2Preconditioner.cpp",
                 "create preconditioner",
                 "Unknown preconditioner \"%s\"", preconditioner.c_str());
  }

  if (preconditioner == "default")
    _name = "diagonal";
}
//-----------------------------------------------------------------------------
Ifpack2Preconditioner::~Ifpack2Preconditioner()
{
}
//-----------------------------------------------------------------------------
void Ifpack2Preconditioner::init(std::shared_ptr<const TpetraMatrix> P)
{
  Ifpack2::Factory prec_factory;

  _prec = prec_factory.create(_name, P->mat());
  Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
  _prec->setParameters(*plist);
  _prec->initialize();
  _prec->compute();
}
//-----------------------------------------------------------------------------
void Ifpack2Preconditioner::set(BelosKrylovSolver& solver)
{
  solver._problem->setRightPrec(_prec);
}
//-----------------------------------------------------------------------------
std::string Ifpack2Preconditioner::str(bool verbose) const
{
  std::stringstream s;

  s << "<Ifpack2Preconditioner>";

  return s.str();
}
//-----------------------------------------------------------------------------
Parameters Ifpack2Preconditioner::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters()("preconditioner"));
  p.rename("ifpack2_preconditioner");

  return p;
}
//-----------------------------------------------------------------------------
#endif

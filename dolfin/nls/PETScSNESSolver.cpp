// Copyright (C) 2012 Patrick E. Farrell
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
//
// First added:  2012-10-13
// Last changed: 2012-10-13

#ifdef HAS_PETSC

#include "PETScSNESSolver.h"

using namespace dolfin;

// Mapping from method string to PETSc
const std::map<std::string, const SNESType> PETScSNESSolver::_methods
  = boost::assign::map_list_of("default",  "")
                              ("ls",          SNESLS)
                              ("tr",          SNESTR)
                              ("nrichardson", SNESNRICHARDSON)
                              ("virs",        SNESVIRS)
                              ("ngmres",      SNESNGMRES)
                              ("qn",          SNESQN)
                              ("ncg",         SNESNCG)
                              ("fas",         SNESFAS)
                              ("ms",          SNESMS);

// Mapping from method string to description
const std::vector<std::pair<std::string, std::string> >
  PETScKrylovSolver::_methods_descr = boost::assign::pair_list_of
    ("default",     "default SNES method")
    ("ls",          "Line search method")
    ("tr",          "Trust region method")
    ("nrichardson", "Richardson nonlinear method (Picard iteration)")
    ("virs",        "Reduced space active set solver method")
    ("ngmres",      "Nonlinear generalised minimum residual method")
    ("qn",          "Limited memory quasi-Newton")
    ("ncg",         "Nonlinear conjugate gradient method")
    ("fas",         "Full Approximation Scheme nonlinear multigrid method")
    ("ms",          "Multistage smoothers");
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
PETScKrylovSolver::methods()
{
  return PETScSNESSolver::_methods_descr;
}
//-----------------------------------------------------------------------------
PETScSNESSolver::PETScSNESSolver(std::string nls_type,
                                 std::string solver_type,
                                 std::string pc_type)
{
  // Check that the requested method is known
  if (_methods.count(nls_type) == 0)
  {
    dolfin_error("PETScSNESSolver.cpp",
                 "create PETSc SNES solver",
                 "Unknown SNES method \"%s\"", method.c_str());
  }
}
//-----------------------------------------------------------------------------
PETScSNESSolver::~PETScSNESSolver()
{
  // Do nothing
}

#endif

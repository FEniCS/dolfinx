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
#include <boost/assign/list_of.hpp>

using namespace dolfin;

// Mapping from method string to PETSc
const std::map<std::string, const SNESType> PETScSNESSolver::_methods
  = boost::assign::map_list_of("default",  "")
                              ("ls",          SNESLS)
                              ("tr",          SNESTR)
                              ("ngmres",      SNESNGMRES);

// These later ones are only available from PETSc 3.3 on, I think
// but at the moment we support PETSc >= 3.2
// so I'm leaving them commented out.
//                              ("nrichardson", SNESNRICHARDSON)
//                              ("virs",        SNESVIRS)
//                              ("qn",          SNESQN)
//                              ("ncg",         SNESNCG)
//                              ("fas",         SNESFAS)
//                              ("ms",          SNESMS);

// Mapping from method string to description
const std::vector<std::pair<std::string, std::string> >
  PETScSNESSolver::_methods_descr = boost::assign::pair_list_of
    ("default",     "default SNES method")
    ("ls",          "Line search method")
    ("tr",          "Trust region method")
    ("ngmres",      "Nonlinear generalised minimum residual method");
//    ("nrichardson", "Richardson nonlinear method (Picard iteration)")
//    ("virs",        "Reduced space active set solver method")
//    ("qn",          "Limited memory quasi-Newton")
//    ("ncg",         "Nonlinear conjugate gradient method")
//    ("fas",         "Full Approximation Scheme nonlinear multigrid method")
//    ("ms",          "Multistage smoothers");

//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
PETScSNESSolver::methods()
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
                 "Unknown SNES method \"%s\"", nls_type.c_str());
  }
}
//-----------------------------------------------------------------------------
PETScSNESSolver::~PETScSNESSolver()
{
  // Do nothing
}

#endif

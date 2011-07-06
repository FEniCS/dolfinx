// Copyright (C) 2006-2009 Garth N. Wells
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
// Modified by Anders Logg, 2006-2010.
//
// First added:  2006-05-31
// Last changed: 2011-03-24

#include <boost/assign/list_of.hpp>
#include <dolfin/common/NoDeleter.h>
#include "uBLASILUPreconditioner.h"
#include "uBLASDummyPreconditioner.h"
#include "uBLASKrylovSolver.h"
#include "KrylovSolver.h"

using namespace dolfin;

const std::set<std::string> uBLASKrylovSolver::solver_types
  = boost::assign::list_of("default")
                          ("cg")
                          ("gmres")
                          ("bicgstab");

//-----------------------------------------------------------------------------
Parameters uBLASKrylovSolver::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters());
  p.rename("ublas_krylov_solver");
  return p;
}
//-----------------------------------------------------------------------------
uBLASKrylovSolver::uBLASKrylovSolver(std::string solver_type,
                                     std::string pc_type)
  : solver_type(solver_type), report(false), parameters_read(false)
{
  // Set parameter values
  parameters = default_parameters();

  // Select and create default preconditioner
  select_preconditioner(pc_type);
}
//-----------------------------------------------------------------------------
uBLASKrylovSolver::uBLASKrylovSolver(uBLASPreconditioner& pc)
  : solver_type("default"), pc(reference_to_no_delete_pointer(pc)),
    report(false), parameters_read(false)
{
  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
uBLASKrylovSolver::uBLASKrylovSolver(std::string solver_type,
                                     uBLASPreconditioner& pc)
  : solver_type(solver_type), pc(reference_to_no_delete_pointer(pc)),
    report(false), parameters_read(false)
{
  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
uBLASKrylovSolver::~uBLASKrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
dolfin::uint uBLASKrylovSolver::solve(const GenericMatrix& A, GenericVector& x,
                                      const GenericVector& b)
{
  return solve(A.down_cast<uBLASMatrix<ublas_sparse_matrix> >(),
               x.down_cast<uBLASVector>(), b.down_cast<uBLASVector>());
}
//-----------------------------------------------------------------------------
dolfin::uint uBLASKrylovSolver::solve(const uBLASMatrix<ublas_dense_matrix>& A,
                                      uBLASVector& x, const uBLASVector& b)
{
  return solve_krylov(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint uBLASKrylovSolver::solve(const uBLASMatrix<ublas_sparse_matrix>& A,
                                      uBLASVector& x, const uBLASVector& b)
{
  return solve_krylov(A, x, b);
}
//-----------------------------------------------------------------------------
dolfin::uint uBLASKrylovSolver::solve(const uBLASKrylovMatrix& A, uBLASVector& x,
                                      const uBLASVector& b)
{
  return solve_krylov(A, x, b);
}
//-----------------------------------------------------------------------------
void uBLASKrylovSolver::select_preconditioner(std::string pc_type)
{
  if(pc_type == "none")
    pc.reset(new uBLASDummyPreconditioner());
  else if (pc_type == "ilu")
    pc.reset(new uBLASILUPreconditioner(parameters));
  else if (pc_type == "default")
    pc.reset(new uBLASILUPreconditioner(parameters));
  else
  {
    warning("Requested preconditioner is not available for uBLAS Krylov solver. Using ILU.");
    pc.reset(new uBLASILUPreconditioner(parameters));
  }
}
//-----------------------------------------------------------------------------
void uBLASKrylovSolver::read_parameters()
{
  // Set tolerances and other parameters
  rtol    = parameters["relative_tolerance"];
  atol    = parameters["absolute_tolerance"];
  div_tol = parameters["divergence_limit"];
  max_it  = parameters["maximum_iterations"];
  restart = parameters("gmres")["restart"];
  report  = parameters["report"];

  // Remember that we have read parameters
  parameters_read = true;
}
//-----------------------------------------------------------------------------

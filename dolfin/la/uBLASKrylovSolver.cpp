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
// Modified by Anders Logg 2006-2012
//
// First added:  2006-05-31
// Last changed: 2012-05-07

#include <boost/assign/list_of.hpp>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/LogStream.h>
#include "uBLASILUPreconditioner.h"
#include "uBLASDummyPreconditioner.h"
#include "uBLASKrylovSolver.h"
#include "KrylovSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
uBLASKrylovSolver::methods()
{
  return boost::assign::pair_list_of
    ("default",  "default Krylov method")
    ("cg",       "Conjugate gradient method")
    ("gmres",    "Generalized minimal residual method")
    ("bicgstab", "Biconjugate gradient stabilized method");
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
uBLASKrylovSolver::preconditioners()
{
  return boost::assign::pair_list_of
    ("default", "default preconditioner")
    ("none",    "No preconditioner")
    ("ilu",     "Incomplete LU factorization");
}
//-----------------------------------------------------------------------------
Parameters uBLASKrylovSolver::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters());
  p.rename("ublas_krylov_solver");
  return p;
}
//-----------------------------------------------------------------------------
uBLASKrylovSolver::uBLASKrylovSolver(std::string method,
                                     std::string preconditioner)
  : method(method), report(false)
{
  // Set parameter values
  parameters = default_parameters();

  // Select and create default preconditioner
  select_preconditioner(preconditioner);
}
//-----------------------------------------------------------------------------
uBLASKrylovSolver::uBLASKrylovSolver(uBLASPreconditioner& pc)
  : method("default"), pc(reference_to_no_delete_pointer(pc)),
    report(false)
{
  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
uBLASKrylovSolver::uBLASKrylovSolver(std::string method,
                                     uBLASPreconditioner& pc)
  : method(method), pc(reference_to_no_delete_pointer(pc)),
    report(false)
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
std::size_t uBLASKrylovSolver::solve(GenericVector& x, const GenericVector& b)
{
  dolfin_assert(A);
  dolfin_assert(P);

  // Try to first use operator as a uBLAS matrix
  if (has_type<const uBLASMatrix<ublas_sparse_matrix> >(*A))
  {
    boost::shared_ptr<const uBLASMatrix<ublas_sparse_matrix> > _A
      = as_type<const uBLASMatrix<ublas_sparse_matrix> >(A);
    boost::shared_ptr<const uBLASMatrix<ublas_sparse_matrix> > _P
      = as_type<const uBLASMatrix<ublas_sparse_matrix> >(P);

    dolfin_assert(_A);
    dolfin_assert(_P);

    return solve_krylov(*_A,
                        as_type<uBLASVector>(x),
                        as_type<const uBLASVector>(b),
                        *_P);
  }

  // If that fails, try to use it as a uBLAS linear operator
  if (has_type<const uBLASLinearOperator>(*A))
  {
    boost::shared_ptr<const uBLASLinearOperator> _A
      =  as_type<const uBLASLinearOperator>(A);
    boost::shared_ptr<const uBLASLinearOperator> _P
      =  as_type<const uBLASLinearOperator>(P);

    dolfin_assert(_A);
    dolfin_assert(_P);

    return solve_krylov(*_A,
                        as_type<uBLASVector>(x),
                        as_type<const uBLASVector>(b),
                        *_P);
  }

  return 0;
}
//-----------------------------------------------------------------------------
std::size_t uBLASKrylovSolver::solve(const GenericLinearOperator& A,
                                     GenericVector& x,
                                     const GenericVector& b)
{
  // Set operator
  boost::shared_ptr<const GenericLinearOperator> _A(&A, NoDeleter());
  set_operator(_A);
  return solve(as_type<uBLASVector>(x), as_type<const uBLASVector>(b));
}
//-----------------------------------------------------------------------------
void uBLASKrylovSolver::select_preconditioner(std::string preconditioner)
{
  if (preconditioner == "none")
    pc.reset(new uBLASDummyPreconditioner());
  else if (preconditioner == "ilu")
    pc.reset(new uBLASILUPreconditioner(parameters));
  else if (preconditioner == "default")
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
}
//-----------------------------------------------------------------------------

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
// Modified by Anders Logg 2006-2011
//
// First added:  2006-05-31
// Last changed: 2011-10-19

#include <boost/assign/list_of.hpp>
#include <dolfin/common/NoDeleter.h>
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
  select_preconditioner(method);
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
dolfin::uint uBLASKrylovSolver::solve(GenericVector& x, const GenericVector& b)
{
  assert(A);
  boost::shared_ptr<const uBLASMatrix<ublas_sparse_matrix> > _A
        = GenericTensor::down_cast<const uBLASMatrix<ublas_sparse_matrix> >(A);
  assert(_A);

  assert(P);
  boost::shared_ptr<const uBLASMatrix<ublas_sparse_matrix> > _P
        = GenericTensor::down_cast<const uBLASMatrix<ublas_sparse_matrix> >(P);
  assert(_P);

  return solve_krylov(*_A, x.down_cast<uBLASVector>(),
                      b.down_cast<uBLASVector>(), *_P);
}
//-----------------------------------------------------------------------------
dolfin::uint uBLASKrylovSolver::solve(const GenericMatrix& A, GenericVector& x,
                                      const GenericVector& b)
{
  // Set operator
  boost::shared_ptr<const GenericMatrix> _A(&A, NoDeleter());
  set_operator(_A);
  return solve(x.down_cast<uBLASVector>(), b.down_cast<uBLASVector>());
}
//-----------------------------------------------------------------------------
dolfin::uint uBLASKrylovSolver::solve(const uBLASKrylovMatrix& A, uBLASVector& x,
                                      const uBLASVector& b)
{
  return solve_krylov(A, x, b, A);
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

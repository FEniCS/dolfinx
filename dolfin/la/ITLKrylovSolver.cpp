// Copyright (C) 2008-2011 Garth N. Wells
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
// Modified by Dag Lindbo 2008
// Modified by Anders Logg 2011
//
// First added:  2008-05-16
// Last changed: 2011-10-19

#ifdef HAS_MTL4

// Order of header files is important
#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/dolfin_log.h>
#include "ITLKrylovSolver.h"
#include "MTL4Matrix.h"
#include "MTL4Vector.h"
#include <boost/assign/list_of.hpp>
#include <boost/numeric/itl/itl.hpp>
#include "KrylovSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
ITLKrylovSolver::methods()
{
  return boost::assign::pair_list_of
    ("default",  "default Krylov method")
    ("cg",       "Conjugate gradient method")
    ("bicgstab", "Biconjugate gradient stabilized method");
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::string, std::string> >
ITLKrylovSolver::preconditioners()
{
  return boost::assign::pair_list_of
    ("default", "default preconditioner")
    ("none",    "No preconditioner")
    ("ilu",     "Incomplete LU factorization")
    ("icc",     "Incomplete Cholesky factorization");
}
//-----------------------------------------------------------------------------
Parameters ITLKrylovSolver::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters());
  p.rename("mtl4_krylov_solver");
  return p;
}
//-----------------------------------------------------------------------------
ITLKrylovSolver::ITLKrylovSolver(std::string method, std::string preconditioner)
                               : method(method), preconditioner(preconditioner)
{
  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
ITLKrylovSolver::~ITLKrylovSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void ITLKrylovSolver::set_operator(const boost::shared_ptr<const GenericMatrix> A)
{
  set_operators(A, A);
}
//-----------------------------------------------------------------------------
void ITLKrylovSolver::set_operators(const boost::shared_ptr<const GenericMatrix> A,
                                    const boost::shared_ptr<const GenericMatrix> P)
{
  this->A = GenericTensor::down_cast<const MTL4Matrix>(A);
  this->A = GenericTensor::down_cast<const MTL4Matrix>(P);
  dolfin_assert(this->A);
  dolfin_assert(this->P);
}
//-----------------------------------------------------------------------------
const GenericMatrix& ITLKrylovSolver::get_operator() const
{
  if (!A)
  {
    dolfin_error("ITLKrylovSolver.cpp",
                 "access operator for ITL Krylov solver",
                 "Operator has not been set");
  }
  return *A;
}
//-----------------------------------------------------------------------------
dolfin::uint ITLKrylovSolver::solve(GenericVector& x, const GenericVector& b)
{
  return solve(x.down_cast<MTL4Vector>(), b.down_cast<MTL4Vector>());
}
//-----------------------------------------------------------------------------
dolfin::uint ITLKrylovSolver::solve(MTL4Vector& x, const MTL4Vector& b)
{
  dolfin_assert(A);
  dolfin_assert(P);

  // Fall back in default method if unknown
  if(method != "cg" && method != "bicgstab" && method != "default")
  {
    warning("Requested ITL Krylov method unknown. Using BiCGStab.");
    method = "bicgstab";
  }

  // Fall back in default preconditioner if unknown
  if(preconditioner != "ilu" &&
     preconditioner != "icc" &&
     preconditioner != "none" &&
     preconditioner != "default")
  {
    warning("Requested ITL preconditioner unknown. Using ILU.");
    preconditioner = "ilu";
  }

  // Set convergence criteria
  itl::basic_iteration<double> iter(b.vec(), parameters["maximum_iterations"],
                                             parameters["relative_tolerance"]);

  // Check vector size
  if( x.size() != b.size() )
  {
    x.resize(b.size());
    x.zero();
  }

  // Solve
  int errno_ = 0;

  // Developers note: the following code is not very elegant.
  // The problem is that ITL are all templates, but DOLFIN selects
  // solvers at runtime. All solvers and preconditioners are instantiated.
  if (preconditioner == "ilu" || preconditioner == "default")
  {
    itl::pc::ilu_0<mtl4_sparse_matrix> pc(P->mat());
    if (method == "cg")
      errno_ = itl::cg(A->mat(), x.vec(), b.vec(), pc, iter);
    else if (method == "bicgstab" || method == "default")
      errno_ = itl::bicgstab(A->mat(), x.vec(), b.vec(), pc, iter);
  }
  else if (preconditioner == "icc")
  {
    itl::pc::ic_0<mtl4_sparse_matrix> pc(P->mat());
    if (method == "cg")
      errno_ = itl::cg(A->mat(), x.vec(), b.vec(), pc, iter);
    else if (method == "bicgstab" || method == "default")
      errno_ = itl::bicgstab(A->mat(), x.vec(), b.vec(), pc, iter);
  }
  else if (preconditioner == "none")
  {
    itl::pc::identity<mtl4_sparse_matrix> pc(P->mat());
    if (method == "cg")
      errno_ = itl::cg(A->mat(), x.vec(), b.vec(), pc, iter);
    else if (method == "bicgstab" || method == "default")
      errno_ = itl::bicgstab(A->mat(), x.vec(), b.vec(), pc, iter);
  }

  // Check exit condition
  if (errno_ == 0)
    log(PROGRESS, "ITLSolver (%s, %s) converged in %d iterations. Resid=%8.2e",
        method.c_str(), preconditioner.c_str(), iter.iterations(), iter.resid());
  else
  {
    bool error_on_nonconvergence = parameters["error_on_nonconvergence"];
    if (error_on_nonconvergence)
    {
      dolfin_error("ITLKrylovSolver.cpp",
                   "solve linear system using ITL Krylov solver",
                   "Solution failed to converge");
    }
    else
    {
      warning("ITL Krylov solver failed to converge (%s, %s)\n\t%d iterations,"
              " Resid=%8.2e",
              method.c_str(),
              preconditioner.c_str(),
              iter.iterations(),
              iter.resid());
    }
  }

  return iter.iterations();
}
//-----------------------------------------------------------------------------
dolfin::uint ITLKrylovSolver::solve(const GenericMatrix& A, GenericVector& x,
                                const GenericVector& b)
{
  boost::shared_ptr<const GenericMatrix> _A(&A, NoDeleter());
  set_operator(_A);
  return solve(x.down_cast<MTL4Vector>(), b.down_cast<MTL4Vector>());
}
//-----------------------------------------------------------------------------
std::string ITLKrylovSolver::str(bool verbose) const
{
  dolfin_not_implemented();
  return std::string();
}
//-----------------------------------------------------------------------------

#endif

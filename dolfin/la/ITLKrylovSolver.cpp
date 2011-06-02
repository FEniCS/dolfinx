// Copyright (C) 2008-2009 Garth N. Wells
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
// Modified by Dag Lindbo, 2008
//
// First added:  2008-05-16
// Last changed: 2011-03-28

#ifdef HAS_MTL4

// Order of header files is important
#include <dolfin/common/NoDeleter.h>
#include <dolfin/log/dolfin_log.h>
#include "ITLKrylovSolver.h"
#include "MTL4Matrix.h"
#include "MTL4Vector.h"
#include <boost/numeric/itl/itl.hpp>
#include "KrylovSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameters ITLKrylovSolver::default_parameters()
{
  Parameters p(KrylovSolver::default_parameters());
  p.rename("mtl4_krylov_solver");
  return p;
}
//-----------------------------------------------------------------------------
ITLKrylovSolver::ITLKrylovSolver(std::string method, std::string pc_type)
                               : method(method), pc_type(pc_type)
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
void ITLKrylovSolver::set_operator(const GenericMatrix& A)
{
  set_operators(A, A);
}
//-----------------------------------------------------------------------------
void ITLKrylovSolver::set_operators(const GenericMatrix& A,
                                    const GenericMatrix& P)
{
  this->AA = reference_to_no_delete_pointer(A);
  this->A = reference_to_no_delete_pointer(A.down_cast<MTL4Matrix>());
  this->P = reference_to_no_delete_pointer(P.down_cast<MTL4Matrix>());
  assert(this->A);
  assert(this->P);
}
//-----------------------------------------------------------------------------
const GenericMatrix& ITLKrylovSolver::get_operator() const
{
  if (!AA)
    error("Operator for linear solver has not been set.");
  return *AA;
}
//-----------------------------------------------------------------------------
dolfin::uint ITLKrylovSolver::solve(GenericVector& x, const GenericVector& b)
{
  check_dimensions(get_operator(), x, b);
  return solve(x.down_cast<MTL4Vector>(), b.down_cast<MTL4Vector>());
}
//-----------------------------------------------------------------------------
dolfin::uint ITLKrylovSolver::solve(MTL4Vector& x, const MTL4Vector& b)
{
  assert(A);
  assert(P);

  // Fall back in default method if unknown
  if(method != "cg" && method != "bicgstab" && method != "default")
  {
    warning("Requested ITL Krylov method unknown. Using BiCGStab.");
    method = "bicgstab";
  }
  // Fall back in default preconditioner if unknown
  if(pc_type != "ilu" && pc_type != "icc" && pc_type != "none" && pc_type != "default")
  {
    warning("Requested ITL preconditioner unknown. Using ILU.");
    pc_type = "ilu";
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
  if(pc_type == "ilu" || pc_type == "default")
  {
    itl::pc::ilu_0<mtl4_sparse_matrix> pc(P->mat());
    if(method == "cg")
      errno_ = itl::cg(A->mat(), x.vec(), b.vec(), pc, iter);
    else if(method == "bicgstab" || method == "default")
      errno_ = itl::bicgstab(A->mat(), x.vec(), b.vec(), pc, iter);
  }
  else if(pc_type == "icc")
  {
    itl::pc::ic_0<mtl4_sparse_matrix> pc(P->mat());
    if(method == "cg")
      errno_ = itl::cg(A->mat(), x.vec(), b.vec(), pc, iter);
    else if(method == "bicgstab" || method == "default")
      errno_ = itl::bicgstab(A->mat(), x.vec(), b.vec(), pc, iter);
  }
  else if(pc_type == "none")
  {
    itl::pc::identity<mtl4_sparse_matrix> pc(P->mat());
    if(method == "cg")
      errno_ = itl::cg(A->mat(), x.vec(), b.vec(), pc, iter);
    else if(method == "bicgstab" || method == "default")
      errno_ = itl::bicgstab(A->mat(), x.vec(), b.vec(), pc, iter);
  }

  // Check exit condition
  if (errno_ == 0)
    log(PROGRESS, "ITLSolver (%s, %s) converged in %d iterations. Resid=%8.2e",
        method.c_str(), pc_type.c_str(), iter.iterations(), iter.resid());
  else
  {
    bool error_on_nonconvergence = parameters["error_on_nonconvergence"];
    if (error_on_nonconvergence)
    {
      error("ITL Krylov solver failed to converge (%s, %s)\n\t%d iterations,"
            " Resid=%8.2e", method.c_str(), pc_type.c_str(), iter.iterations(), iter.resid());
    }
    else
    {
      warning("ITL Krylov solver failed to converge (%s, %s)\n\t%d iterations,"
              " Resid=%8.2e", method.c_str(), pc_type.c_str(), iter.iterations(), iter.resid());
    }
  }

  return iter.iterations();
}
//-----------------------------------------------------------------------------
dolfin::uint ITLKrylovSolver::solve(const GenericMatrix& A, GenericVector& x,
                                const GenericVector& b)
{
  check_dimensions(A, x, b);
  set_operator(A);
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

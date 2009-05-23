// Copyright (C) 2006-2009 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2006-2008.
//
// First added:  2006-05-31
// Last changed: 2009-05-23

#include <boost/assign/list_of.hpp>
#include "uBLASILUPreconditioner.h"
#include "uBLASDummyPreconditioner.h"
#include "uBLASKrylovSolver.h"

using namespace dolfin;

const std::set<std::string> uBLASKrylovSolver::solver_types 
    = boost::assign::list_of("default")
                            ("cg")
                            ("gmres")
                            ("bicgstab"); 

//-----------------------------------------------------------------------------
uBLASKrylovSolver::uBLASKrylovSolver(std::string solver_type, std::string pc_type)
  : solver_type(solver_type), pc_user(false), report(false), 
    parameters_read(false)
{
  // Select and create default preconditioner
  select_preconditioner(pc_type);
}
//-----------------------------------------------------------------------------
uBLASKrylovSolver::uBLASKrylovSolver(uBLASPreconditioner& pc)
  : solver_type("default"), pc(&pc), pc_user(true), report(false), 
    parameters_read(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBLASKrylovSolver::uBLASKrylovSolver(std::string solver_type, uBLASPreconditioner& pc)
  : solver_type(solver_type), pc(&pc), pc_user(true), report(false), 
    parameters_read(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
uBLASKrylovSolver::~uBLASKrylovSolver()
{
  // Delete preconditioner if it was not created by user
  if( !pc_user )
    delete pc;
}
//-----------------------------------------------------------------------------
dolfin::uint uBLASKrylovSolver::solve(const GenericMatrix& A, GenericVector& x,
                                      const GenericVector& b)
{
  return solve(A.down_cast<uBLASMatrix<ublas_sparse_matrix> >(),
               x.down_cast<uBLASVector>(),  b.down_cast<uBLASVector>());
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
    pc = new uBLASDummyPreconditioner();
  else if (pc_type == "ilu")
    pc = new uBLASILUPreconditioner();
  else if (pc_type == "default")
    pc = new uBLASILUPreconditioner();
  else
  {
    warning("Requested preconditioner is not available for uBLAS Krylov solver. Using ILU.");
    pc = new uBLASILUPreconditioner();
  }
}
//-----------------------------------------------------------------------------
void uBLASKrylovSolver::read_parameters()
{
  // Set tolerances and other parameters
  rtol    = get("Krylov relative tolerance");
  atol    = get("Krylov absolute tolerance");
  div_tol = get("Krylov divergence limit");
  max_it  = get("Krylov maximum iterations");
  restart = get("Krylov GMRES restart");
  report  = get("Krylov report");

  // Remember that we have read parameters
  parameters_read = true;
}
//-----------------------------------------------------------------------------

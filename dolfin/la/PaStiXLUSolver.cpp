// Copyright (C) 2011 Garth N. Wells
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
// First added:  2011-10-19
// Last changed:

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

// Necessary since pastix.h does not include it
#include <stdint.h>

#include "dolfin/common/MPI.h"
#include "dolfin/common/NoDeleter.h"
#include "dolfin/common/types.h"
#include "dolfin/log/log.h"
#include "dolfin/parameter/GlobalParameters.h"
#include "GenericVector.h"
#include "SparsityPattern.h"
#include "GenericMatrix.h"
#include "LUSolver.h"
#include "STLMatrix.h"
#include "PaStiXLUSolver.h"

#ifdef HAS_PASTIX

extern "C"
{
#include <pastix.h>
}

using namespace dolfin;

//-----------------------------------------------------------------------------
Parameters PaStiXLUSolver::default_parameters()
{
  Parameters p(LUSolver::default_parameters());
  p.rename("pastix_lu_solver");

  // Number of threads per MPI process
  p.add<std::size_t>("num_threads");

  // Thread mode (see PaStiX user documentation)
  const std::set<std::string> thread_modes = {"multiple", "single", "funnel"};
  p.add<std::string>("thread_mode", thread_modes);

  // Min/max block size for BLAS. This parameters can have a significant
  // effect on performance. Best settings depends on systems and BLAS
  // implementation.
  p.add("min_block_size", 180);
  p.add("max_block_size", 340);

  // Check matrix for consistency
  p.add("check_matrix", false);

  // Renumber
  p.add("renumber", true);

  return p;
}
//-----------------------------------------------------------------------------
PaStiXLUSolver::PaStiXLUSolver(std::shared_ptr<const STLMatrix> A) : A(A)
{
  // Set parameter values
  parameters = default_parameters();
}
//-----------------------------------------------------------------------------
PaStiXLUSolver::~PaStiXLUSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t PaStiXLUSolver::solve(GenericVector& x, const GenericVector& b)
{
  dolfin_assert(A);

  // Get block size from STLMatrix
  const std::size_t block_size = A->block_size();

  // MPI communicator
  MPI_Comm mpi_comm = A->mpi_comm();

  // Initialise PaStiX parameters
  pastix_int_t iparm[IPARM_SIZE];
  double dparm[DPARM_SIZE];
  for (int i = 0; i < IPARM_SIZE; i++)
    iparm[i] = 0;
  for (int i = 0; i < DPARM_SIZE; i++)
    dparm[i] = 0;

  // Set default parameters
  pastix_initParam(iparm, dparm);

  // Set LU or Cholesky depending on operator symmetry
  const bool symmetric = parameters["symmetric"];
  if (symmetric)
  {
    iparm[IPARM_SYM] = API_SYM_YES;
    //iparm[IPARM_FACTORIZATION] = API_FACT_LDLT;
    iparm[IPARM_FACTORIZATION] = API_FACT_LLT;
  }
  else
  {
    iparm[IPARM_SYM] = API_SYM_NO;
    iparm[IPARM_FACTORIZATION] = API_FACT_LU;
  }

  // Number of dofs per node (block size)
  iparm[IPARM_DOF_NBR] = block_size;

  // Set BLAS block sizes (affects performance)
  const std::size_t min_block_size = parameters["min_block_size"];
  iparm[IPARM_MIN_BLOCKSIZE] = min_block_size;
  const std::size_t max_block_size = parameters["max_block_size"];
  iparm[IPARM_MAX_BLOCKSIZE] = max_block_size;
  //iparm[IPARM_ABS] = API_YES;

  // Get matrix data in compressed sparse column format (C indexing)
  std::vector<double> vals;
  std::vector<pastix_int_t> rows, col_ptr, local_to_global_cols;
  A->csc(vals, rows, col_ptr, local_to_global_cols, true, symmetric);

  // Copy local-to-global
  const std::vector<pastix_int_t> local_to_global_cols_ref
    = local_to_global_cols;

  // Convert to base 1
  for (std::size_t i = 0;  i < rows.size(); ++i)
    rows[i] += 1;
  for (std::size_t i = 0;  i < col_ptr.size(); ++i)
    col_ptr[i] += 1;
  for (std::size_t i = 0;  i < local_to_global_cols.size(); ++i)
    local_to_global_cols[i] += 1;

  dolfin_assert(local_to_global_cols.size() > 0);

  // Pointers to data structures
  pastix_int_t* _col_ptr = col_ptr.data();
  pastix_int_t* _rows = rows.data();
  pastix_int_t* _local_to_global_cols = local_to_global_cols.data();

  // Graph (matrix) distributed
  iparm[IPARM_GRAPHDIST] = API_YES;

  // Matrix size
  const pastix_int_t n = (col_ptr.size() - 1);

  // Check matrix
  if (parameters["check_matrix"])
  {
    double* _vals = vals.data();
    d_pastix_checkMatrix(mpi_comm, API_VERBOSE_YES, iparm[IPARM_SYM], API_NO,
  		                   n, &_col_ptr, &_rows, &_vals,
                         &_local_to_global_cols, block_size);
  }
  else
    iparm[IPARM_MATRIX_VERIFICATION] = API_NO;

  // PaStiX object
  pastix_data_t* pastix_data = NULL;

  // Number of threads per MPI process
  if (parameters["num_threads"].is_set())
  {
    iparm[IPARM_THREAD_NBR]
      = std::max((std::size_t) 1, (std::size_t) parameters["num_threads"]);
  }
  else
  {
    iparm[IPARM_THREAD_NBR] = std::max((std::size_t) 1,
                            (std::size_t) dolfin::parameters["num_threads"]);
  }

  // PaStiX threading mode
  if (parameters["thread_mode"].is_set())
  {
    const std::string thread_mode = parameters["thread_mode"];
    if (thread_mode == "multiple")
      iparm[IPARM_THREAD_COMM_MODE] = API_THREAD_MULTIPLE;
    else if (thread_mode == "single")
      iparm[IPARM_THREAD_COMM_MODE] = API_THREAD_COMM_ONE;
    else if (thread_mode == "funnel")
      iparm[IPARM_THREAD_COMM_MODE] = API_THREAD_FUNNELED;
    else
    {
      dolfin_error("PaStiXLUSolver.cpp",
                   "set PaStiX thread mode",
                   "Unknown mode \"%s\"", thread_mode.c_str());
    }
  }

  // User-supplied RHS
  iparm[IPARM_RHS_MAKING] = API_RHS_B;

  // Level of verbosity
  if (parameters["verbose"])
    iparm[IPARM_VERBOSE] = API_VERBOSE_YES;
  else
    iparm[IPARM_VERBOSE] = API_VERBOSE_NO;

  // Allocate space for solver
  dolfin_assert(local_to_global_cols.size() > 0);
  std::vector<pastix_int_t> perm(local_to_global_cols.size());
  std::vector<pastix_int_t> invp(local_to_global_cols.size());

  // Renumbering
  const bool renumber = parameters["renumber"];
  if (!renumber)
  {
    iparm[IPARM_ORDERING] = API_ORDER_PERSONAL;
    iparm[IPARM_LEVEL_OF_FILL] = -1;
    iparm[IPARM_AMALGAMATION_LEVEL]  = 10;
    for (std::size_t i = 0; i < local_to_global_cols.size(); ++i)
    {
      perm[i] = i + 1;
      std::copy(perm.begin(), perm.end(), invp.begin());
    }
  }

  // Number of RHS vectors
  const pastix_int_t nrhs = 1;

  // Re-order
  iparm[IPARM_START_TASK] = API_TASK_ORDERING;
  iparm[IPARM_END_TASK]   = API_TASK_BLEND;
  d_dpastix(&pastix_data, mpi_comm, n, _col_ptr, _rows, vals.data(),
            _local_to_global_cols, perm.data(), invp.data(), NULL, nrhs,
            iparm, dparm);

  // Factorise
  iparm[IPARM_START_TASK] = API_TASK_NUMFACT;
  iparm[IPARM_END_TASK]   = API_TASK_NUMFACT;
  d_dpastix(&pastix_data, mpi_comm, n, _col_ptr, _rows, vals.data(),
            _local_to_global_cols, perm.data(), invp.data(), NULL, nrhs,
            iparm, dparm);

  // Get RHS data for this process
  std::vector<double> _b;
  std::vector<dolfin::la_index> idx(block_size*n);
  dolfin_assert((int) local_to_global_cols_ref.size() ==  n);
  for (std::size_t i = 0; i < local_to_global_cols_ref.size(); ++i)
    for (std::size_t j = 0; j < block_size; ++j)
      idx[i*block_size + j] = local_to_global_cols_ref[i]*block_size + j;
  b.gather(_b, idx);

  // Solve
  iparm[IPARM_START_TASK] = API_TASK_SOLVE;
  iparm[IPARM_END_TASK] = API_TASK_SOLVE;
  d_dpastix(&pastix_data, mpi_comm, n, _col_ptr, _rows, vals.data(),
            _local_to_global_cols, perm.data(), invp.data(),
            _b.data(), nrhs, iparm, dparm);

  // Distribute solution
  x.init(mpi_comm, b.local_range());
  x.set(_b.data(), idx.size(), idx.data());
  x.apply("insert");

  // Clean up
  iparm[IPARM_START_TASK] = API_TASK_CLEAN;
  iparm[IPARM_END_TASK] = API_TASK_CLEAN;
  d_dpastix(&pastix_data, mpi_comm, n, NULL, NULL, NULL,
            _local_to_global_cols,
            perm.data(), invp.data(),
            _b.data(), nrhs, iparm, dparm);

  return 1;
}
//-----------------------------------------------------------------------------
#endif

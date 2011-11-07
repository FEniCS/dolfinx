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

#include <numeric>
#include <string>
#include <vector>

#include "dolfin/common/Array.h"
#include "dolfin/common/MPI.h"
#include "dolfin/common/NoDeleter.h"
#include "dolfin/common/types.h"
#include "dolfin/log/dolfin_log.h"
#include "GenericVector.h"
#include "SparsityPattern.h"
#include "GenericMatrix.h"
#include "STLMatrix.h"
#include "PaStiXLUSolver.h"

#ifdef HAS_PASTIX

extern "C"
{
#include <pastix.h>
}


using namespace dolfin;

//-----------------------------------------------------------------------------
PaStiXLUSolver::PaStiXLUSolver(const STLMatrix& A)
  : A(reference_to_no_delete_pointer(A)), id(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PaStiXLUSolver::PaStiXLUSolver(boost::shared_ptr<const STLMatrix> A)
  : A(A), id(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
unsigned int PaStiXLUSolver::solve(GenericVector& x, const GenericVector& b)
{
  assert(A);

  MPI_Comm mpi_comm = MPI_COMM_WORLD;

  int    iparm[IPARM_SIZE];
  double dparm[DPARM_SIZE];
  for (int i = 0; i < IPARM_SIZE; i++)
    iparm[i] = 0;
  for (int i = 0; i < DPARM_SIZE; i++)
    dparm[i] = 0;

  // Set default parameters
  pastix_initParam(iparm, dparm);

  // PaStiX object
  pastix_data_t* pastix_data = NULL;

  // Matrix data
  std::vector<double> vals;
  std::vector<uint> cols, row_ptr, local_to_global_rows;
  A->csr(vals, cols, row_ptr, local_to_global_rows, false);

  int* _row_ptr = reinterpret_cast<int*>(&row_ptr[0]);
  int* _cols = reinterpret_cast<int*>(&cols[0]);
  int* _local_to_global_rows = reinterpret_cast<int*>(&local_to_global_rows[0]);
  double* _vals = &vals[0];

  const uint n = row_ptr.size() - 1;

  // Check matrix
  d_pastix_checkMatrix(mpi_comm, API_VERBOSE_YES,
		                   API_SYM_YES,  API_YES,
		                   n, &_row_ptr, &_cols, &_vals, &_local_to_global_rows, 1);

  // Number of threads per MPI process
  iparm[IPARM_THREAD_NBR] = 1;

  // User-supplied RHS
  iparm[IPARM_RHS_MAKING] = API_RHS_B;

  // Level of verbosity
  iparm[IPARM_VERBOSE] = API_VERBOSE_YES;

  // LU or Cholesky
  iparm[IPARM_SYM] = API_SYM_YES;
  iparm[IPARM_FACTORIZATION] = API_FACT_LLT;
  //iparm[IPARM_SYM] = API_SYM_NO;
  //iparm[IPARM_FACTORIZATION] = API_FACT_LU;

  // Graph (matrix) is distributed
  iparm[IPARM_GRAPHDIST] = API_YES;

  Array<int> perm(local_to_global_rows.size());
  Array<int> invp(local_to_global_rows.size());

  const int nrhs = 1;

  // Re-order
  iparm[IPARM_START_TASK] = API_TASK_ORDERING;
  iparm[IPARM_END_TASK]   = API_TASK_BLEND;
  d_dpastix(&pastix_data, mpi_comm, n, _row_ptr, _cols, _vals,
            _local_to_global_rows,
            perm.data().get(), invp.data().get(),
            NULL, nrhs, iparm, dparm);

  // Factorise
  iparm[IPARM_START_TASK] = API_TASK_NUMFACT;
  iparm[IPARM_END_TASK]   = API_TASK_NUMFACT;
  d_dpastix(&pastix_data, mpi_comm, n, _row_ptr, _cols, _vals,
            _local_to_global_rows,
            perm.data().get(), invp.data().get(),
            NULL, nrhs, iparm, dparm);

  // Get local (to process) dofs
  const uint ncol2 = pastix_getLocalNodeNbr(&pastix_data);
  Array<uint> solver_local_to_global(ncol2);
  int* _loc2glob = reinterpret_cast<int*>(solver_local_to_global.data().get());
  pastix_getLocalNodeLst(&pastix_data, _loc2glob) ;

  // Perform shift
  for (uint i = 0; i < ncol2; ++i)
    _loc2glob[i]--;

  // Get RHS data for this process
  Array<double> _b(ncol2);
  b.gather(_b, solver_local_to_global);
  double* b_ptr = _b.data().get();

  // Solve
  iparm[IPARM_START_TASK] = API_TASK_SOLVE;
  iparm[IPARM_END_TASK] = API_TASK_SOLVE;
  d_dpastix(&pastix_data, mpi_comm, n, NULL, NULL, NULL,
            _local_to_global_rows,
            perm.data().get(), invp.data().get(),
            b_ptr, nrhs, iparm, dparm);

  // FIXME: Use pastix getLocalUnknownNbr?

  // Distribute solution
  assert(b.size() == x.size());
  x.set(_b.data().get(), ncol2, solver_local_to_global.data().get());
  x.apply("insert");

  // Clean up
  iparm[IPARM_START_TASK] = API_TASK_CLEAN;
  iparm[IPARM_END_TASK] = API_TASK_CLEAN;
  d_dpastix(&pastix_data, mpi_comm, n, NULL, NULL, NULL, NULL,
            perm.data().get(), invp.data().get(),
            NULL, nrhs, iparm, dparm);

  return 1;
}
//-----------------------------------------------------------------------------
PaStiXLUSolver::~PaStiXLUSolver()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
#endif


#include "superlu.h"
#include "superlu_ddefs.h"
#include "superlu_sdefs.h"
#include "superlu_zdefs.h"
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <iostream>

using namespace dolfinx;

template <typename T>
int superlu_solver(MPI_Comm comm, const la::MatrixCSR<T>& Amat,
                   const la::Vector<T>& bvec, la::Vector<T>& uvec, bool verbose)
{
  int size = dolfinx::MPI::size(comm);

  int nprow = 1;
  int npcol = size;

  gridinfo_t grid;
  superlu_gridinit(MPI_COMM_WORLD, nprow, npcol, &grid);

  // Global size
  int m = Amat.index_map(0)->size_global();
  int n = Amat.index_map(1)->size_global();
  if (m != n)
    throw std::runtime_error("Can't solve non-square system");

  // Number of local rows
  int m_loc = Amat.num_owned_rows();

  // First row
  int first_row = Amat.index_map(0)->local_range()[0];

  // Local number of non-zeros
  int nnz_loc = Amat.row_ptr()[m_loc];
  std::vector<int_t> cols(nnz_loc);
  std::vector<int_t> rowptr(m_loc + 1);

  // Copy row_ptr from int64
  std::copy(Amat.row_ptr().begin(),
            std::next(Amat.row_ptr().begin(), m_loc + 1), rowptr.begin());

  // Convert local to global indices (and cast to int_t)
  std::vector<std::int64_t> global_col_indices(
      Amat.index_map(1)->global_indices());
  std::transform(Amat.cols().begin(), std::next(Amat.cols().begin(), nnz_loc),
                 cols.begin(),
                 [&](std::int64_t local_index)
                 { return global_col_indices[local_index]; });

  SuperMatrix A;
  auto Amatdata = const_cast<T*>(Amat.values().data());
  if constexpr (std::is_same_v<T, double>)
  {
    dCreate_CompRowLoc_Matrix_dist(&A, m, n, nnz_loc, m_loc, first_row,
                                   Amatdata, cols.data(), rowptr.data(),
                                   SLU_NR_loc, SLU_D, SLU_GE);
  }
  else if constexpr (std::is_same_v<T, float>)
  {
    sCreate_CompRowLoc_Matrix_dist(&A, m, n, nnz_loc, m_loc, first_row,
                                   Amatdata, cols.data(), rowptr.data(),
                                   SLU_NR_loc, SLU_S, SLU_GE);
  }
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    zCreate_CompRowLoc_Matrix_dist(&A, m, n, nnz_loc, m_loc, first_row,
                                   reinterpret_cast<doublecomplex*>(Amatdata),
                                   cols.data(), rowptr.data(), SLU_NR_loc,
                                   SLU_Z, SLU_GE);
  }

  // RHS
  int ldb = m_loc;
  int nrhs = 1;

  superlu_dist_options_t options;
  set_default_options_dist(&options);
  options.DiagInv = YES;
  options.ReplaceTinyPivot = YES;
  if (!verbose)
    options.PrintStat = NO;

  int info = 0;
  SuperLUStat_t stat;
  PStatInit(&stat);

  // Copy b to u (SuperLU replaces RHS with solution)
  std::copy(bvec.array().begin(), std::next(bvec.array().begin(), m_loc),
            uvec.mutable_array().begin());

  if constexpr (std::is_same_v<T, double>)
  {
    std::vector<T> berr(nrhs);
    dScalePermstruct_t ScalePermstruct;
    dLUstruct_t LUstruct;
    dScalePermstructInit(m, n, &ScalePermstruct);
    dLUstructInit(n, &LUstruct);
    dSOLVEstruct_t SOLVEstruct;

    pdgssvx(&options, &A, &ScalePermstruct, uvec.mutable_array().data(), ldb,
            nrhs, &grid, &LUstruct, &SOLVEstruct, berr.data(), &stat, &info);

    dScalePermstructFree(&ScalePermstruct);
    dLUstructFree(&LUstruct);
    dSolveFinalize(&options, &SOLVEstruct);
  }
  else if constexpr (std::is_same_v<T, float>)
  {
    std::vector<T> berr(nrhs);
    sScalePermstruct_t ScalePermstruct;
    sLUstruct_t LUstruct;
    sScalePermstructInit(m, n, &ScalePermstruct);
    sLUstructInit(n, &LUstruct);
    sSOLVEstruct_t SOLVEstruct;

    psgssvx(&options, &A, &ScalePermstruct, uvec.mutable_array().data(), ldb,
            nrhs, &grid, &LUstruct, &SOLVEstruct, berr.data(), &stat, &info);

    sSolveFinalize(&options, &SOLVEstruct);
    sLUstructFree(&LUstruct);
    sScalePermstructFree(&ScalePermstruct);
  }
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    std::vector<double> berr(nrhs);
    zScalePermstruct_t ScalePermstruct;
    zLUstruct_t LUstruct;
    zScalePermstructInit(m, n, &ScalePermstruct);
    zLUstructInit(n, &LUstruct);
    zSOLVEstruct_t SOLVEstruct;

    pzgssvx(&options, &A, &ScalePermstruct,
            reinterpret_cast<doublecomplex*>(uvec.mutable_array().data()), ldb,
            nrhs, &grid, &LUstruct, &SOLVEstruct, berr.data(), &stat, &info);

    zScalePermstructFree(&ScalePermstruct);
    zLUstructFree(&LUstruct);
    zSolveFinalize(&options, &SOLVEstruct);
  }
  Destroy_SuperMatrix_Store_dist(&A);

  if (info)
  {
    std::cout << "ERROR: INFO = " << info << " returned from p*gssvx()\n"
              << std::flush;
  }

  if (verbose)
    PStatPrint(&options, &stat, &grid);
  PStatFree(&stat);

  superlu_gridexit(&grid);

  // Update ghosts in u
  uvec.scatter_fwd();
  return info;
}

// Explicit instantiation
template int superlu_solver(MPI_Comm, const la::MatrixCSR<double>&,
                            const la::Vector<double>&, la::Vector<double>&,
                            bool);

template int superlu_solver(MPI_Comm, const la::MatrixCSR<float>&,
                            const la::Vector<float>&, la::Vector<float>&, bool);

template int superlu_solver(MPI_Comm,
                            const la::MatrixCSR<std::complex<double>>&,
                            const la::Vector<std::complex<double>>&,
                            la::Vector<std::complex<double>>&, bool);

template <typename T>
SuperLUSolver<T>::SuperLUSolver(MPI_Comm comm, bool verbose)
    : _comm(comm), _verbose(verbose)
{
  int size = dolfinx::MPI::size(comm);

  int nprow = size;
  int npcol = 1;
  _grid = std::make_shared<gridinfo_t>();
  superlu_gridinit(MPI_COMM_WORLD, nprow, npcol, (gridinfo_t*)_grid.get());
}

template <typename T>
SuperLUSolver<T>::~SuperLUSolver()
{
  if (_A)
    Destroy_SuperMatrix_Store_dist((SuperMatrix*)_A.get());
  superlu_gridexit((gridinfo_t*)_grid.get());
}

template <typename T>
void SuperLUSolver<T>::set_operator(const la::MatrixCSR<T>& Amat)
{
  spdlog::info("Set operator");
  // Global size
  m = Amat.index_map(0)->size_global();
  int n = Amat.index_map(1)->size_global();
  if (m != n)
    throw std::runtime_error("Can't solve non-square system");

  // Number of local rows
  m_loc = Amat.num_owned_rows();

  // First row
  int first_row = Amat.index_map(0)->local_range()[0];

  // Local number of non-zeros
  int nnz_loc = Amat.row_ptr()[m_loc];
  cols.resize(nnz_loc);
  rowptr.resize(m_loc + 1);

  // Copy row_ptr from int64
  std::copy(Amat.row_ptr().begin(),
            std::next(Amat.row_ptr().begin(), m_loc + 1), rowptr.begin());

  // Convert local to global indices (and cast to int_t)
  std::vector<std::int64_t> global_col_indices(
      Amat.index_map(1)->global_indices());
  std::transform(Amat.cols().begin(), std::next(Amat.cols().begin(), nnz_loc),
                 cols.begin(),
                 [&](std::int64_t local_index)
                 { return global_col_indices[local_index]; });

  _A = std::make_shared<SuperMatrix>();
  auto Amatdata = const_cast<T*>(Amat.values().data());
  if constexpr (std::is_same_v<T, double>)
  {
    dCreate_CompRowLoc_Matrix_dist((SuperMatrix*)_A.get(), m, n, nnz_loc, m_loc,
                                   first_row, Amatdata, cols.data(),
                                   rowptr.data(), SLU_NR_loc, SLU_D, SLU_GE);
  }
  spdlog::info("Done set operator");
}

template <typename T>
int SuperLUSolver<T>::solve(const la::Vector<T>& bvec, la::Vector<T>& uvec)
{
  // RHS
  int ldb = m_loc;
  int nrhs = 1;

  superlu_dist_options_t options;
  set_default_options_dist(&options);
  options.DiagInv = YES;
  options.ReplaceTinyPivot = YES;
  if (!_verbose)
    options.PrintStat = NO;

  int info = 0;
  SuperLUStat_t stat;
  PStatInit(&stat);

  // Copy b to u (SuperLU replaces RHS with solution)
  std::copy(bvec.array().begin(), std::next(bvec.array().begin(), m_loc),
            uvec.mutable_array().begin());

  if constexpr (std::is_same_v<T, double>)
  {
    spdlog::info("Start solve");
    std::vector<T> berr(nrhs);
    dScalePermstruct_t ScalePermstruct;
    dLUstruct_t LUstruct;
    dScalePermstructInit(m, m, &ScalePermstruct);
    dLUstructInit(m, &LUstruct);
    dSOLVEstruct_t SOLVEstruct;

    spdlog::info("Call pdgssvx");
    pdgssvx(&options, (SuperMatrix*)_A.get(), &ScalePermstruct,
            uvec.mutable_array().data(), ldb, nrhs, (gridinfo_t*)_grid.get(),
            &LUstruct, &SOLVEstruct, berr.data(), &stat, &info);

    dScalePermstructFree(&ScalePermstruct);
    dLUstructFree(&LUstruct);
    spdlog::info("Solve finalize");
    dSolveFinalize(&options, &SOLVEstruct);
  }
  if (info)
  {
    std::cout << "ERROR: INFO = " << info << " returned from p*gssvx()\n"
              << std::flush;
  }

  if (_verbose)
    PStatPrint(&options, &stat, (gridinfo_t*)_grid.get());
  PStatFree(&stat);

  // Update ghosts in u
  uvec.scatter_fwd();
  return info;
}

template class SuperLUSolver<double>;

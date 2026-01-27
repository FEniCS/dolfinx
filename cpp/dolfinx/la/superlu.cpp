// Copyright (C) 2026 Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_SUPERLU_DIST

#include "superlu.h"
extern "C"
{
#include "superlu_ddefs.h"
#include "superlu_sdefs.h"
#include "superlu_zdefs.h"
}
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <iostream>

using namespace dolfinx;
using namespace dolfinx::la;

template <typename T>
SuperLUSolver<T>::SuperLUSolver(std::shared_ptr<const la::MatrixCSR<T>> Amat,
                                bool verbose)
    : _grid(nullptr), _A(nullptr), _Amat(Amat), _verbose(verbose)
{
  int size = dolfinx::MPI::size(Amat->comm());

  int nprow = size;
  int npcol = 1;
  _grid = new gridinfo_t; // Allocate memory for grid
  superlu_gridinit(Amat->comm(), nprow, npcol, (gridinfo_t*)_grid);
  _A = new SuperMatrix; // Allocate memory for SuperMatrix
  set_operator(*Amat);
}
//---------------------------------------------------------------------------------------
template <typename T>
SuperLUSolver<T>::~SuperLUSolver()
{
  Destroy_SuperMatrix_Store_dist((SuperMatrix*)_A);
  delete (SuperMatrix*)_A; // Free SuperMatrix
  superlu_gridexit((gridinfo_t*)_grid);
  delete (gridinfo_t*)_grid; // Free memory for grid
}
//---------------------------------------------------------------------------------------
template <typename T>
void SuperLUSolver<T>::set_operator(const la::MatrixCSR<T>& Amat)
{
  spdlog::info("Set operator");
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

  auto Amatdata = const_cast<T*>(Amat.values().data());
  if constexpr (std::is_same_v<T, double>)
  {
    dCreate_CompRowLoc_Matrix_dist((SuperMatrix*)_A, m, n, nnz_loc, m_loc,
                                   first_row, Amatdata, cols.data(),
                                   rowptr.data(), SLU_NR_loc, SLU_D, SLU_GE);
  }
  if constexpr (std::is_same_v<T, float>)
  {
    sCreate_CompRowLoc_Matrix_dist((SuperMatrix*)_A, m, n, nnz_loc, m_loc,
                                   first_row, Amatdata, cols.data(),
                                   rowptr.data(), SLU_NR_loc, SLU_S, SLU_GE);
  }
  if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    zCreate_CompRowLoc_Matrix_dist(
        (SuperMatrix*)_A, m, n, nnz_loc, m_loc, first_row,
        reinterpret_cast<doublecomplex*>(Amatdata), cols.data(), rowptr.data(),
        SLU_NR_loc, SLU_Z, SLU_GE);
  }
  spdlog::info("Done set operator");
}
//---------------------------------------------------------------------------------------
template <typename T>
int SuperLUSolver<T>::solve(const la::Vector<T>& bvec, la::Vector<T>& uvec)
{
  assert(_A);
  int m = _Amat->index_map(0)->size_global();
  int m_loc = _Amat->num_owned_rows();

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
            uvec.array().begin());

  if constexpr (std::is_same_v<T, double>)
  {
    std::vector<T> berr(nrhs);
    spdlog::info("Start solve [float64]");
    dScalePermstruct_t ScalePermstruct;
    dLUstruct_t LUstruct;
    dScalePermstructInit(m, m, &ScalePermstruct);
    dLUstructInit(m, &LUstruct);
    dSOLVEstruct_t SOLVEstruct;

    spdlog::info("Call pdgssvx");
    pdgssvx(&options, (SuperMatrix*)_A, &ScalePermstruct, uvec.array().data(),
            ldb, nrhs, (gridinfo_t*)_grid, &LUstruct, &SOLVEstruct, berr.data(),
            &stat, &info);

    dScalePermstructFree(&ScalePermstruct);
    dLUstructFree(&LUstruct);
    spdlog::info("Solve finalize");
    dSolveFinalize(&options, &SOLVEstruct);
  }
  else if constexpr (std::is_same_v<T, float>)
  {
    std::vector<T> berr(nrhs);
    spdlog::info("Start solve [float32]");
    sScalePermstruct_t ScalePermstruct;
    sLUstruct_t LUstruct;
    sScalePermstructInit(m, m, &ScalePermstruct);
    sLUstructInit(m, &LUstruct);
    sSOLVEstruct_t SOLVEstruct;

    spdlog::info("Call psgssvx");
    psgssvx(&options, (SuperMatrix*)_A, &ScalePermstruct, uvec.array().data(),
            ldb, nrhs, (gridinfo_t*)_grid, &LUstruct, &SOLVEstruct, berr.data(),
            &stat, &info);

    spdlog::info("Solve finalize");
    sSolveFinalize(&options, &SOLVEstruct);
    sScalePermstructFree(&ScalePermstruct);
    sLUstructFree(&LUstruct);
  }
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    std::vector<double> berr(nrhs);
    spdlog::info("Start solve [complex]");
    zScalePermstruct_t ScalePermstruct;
    zLUstruct_t LUstruct;
    zScalePermstructInit(m, m, &ScalePermstruct);
    zLUstructInit(m, &LUstruct);
    zSOLVEstruct_t SOLVEstruct;

    spdlog::info("Call pzgssvx");
    pzgssvx(&options, (SuperMatrix*)_A, &ScalePermstruct,
            reinterpret_cast<doublecomplex*>(uvec.array().data()), ldb, nrhs,
            (gridinfo_t*)_grid, &LUstruct, &SOLVEstruct, berr.data(), &stat,
            &info);

    zScalePermstructFree(&ScalePermstruct);
    zLUstructFree(&LUstruct);
    spdlog::info("Solve finalize");
    zSolveFinalize(&options, &SOLVEstruct);
  }
  else
    throw std::runtime_error("Invalid scalar type");

  if (info)
  {
    std::cout << "ERROR: INFO = " << info << " returned from p*gssvx()\n"
              << std::flush;
  }

  if (_verbose)
    PStatPrint(&options, &stat, (gridinfo_t*)_grid);
  PStatFree(&stat);

  // Update ghosts in u
  uvec.scatter_fwd();
  return info;
}
//---------------------------------------------------------------------------------------
template class la::SuperLUSolver<double>;
template class la::SuperLUSolver<float>;
template class la::SuperLUSolver<std::complex<double>>;

#endif

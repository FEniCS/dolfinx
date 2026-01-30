// Copyright (C) 2026 Jack S. Hale, Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_SUPERLU_DIST

#include "superlu_dist.h"
extern "C"
{
#include "superlu_ddefs.h"
#include "superlu_sdefs.h"
#include "superlu_zdefs.h"
}
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <iostream>

// Trick for declaring anonymous typedef structs from SuperLU_DIST
struct dolfinx::la::SuperLUDistStructs::SuperMatrix : public ::SuperMatrix
{
};

struct dolfinx::la::SuperLUDistStructs::gridinfo_t : public ::gridinfo_t
{
};

/// Struct holding vector of type int_t
struct dolfinx::la::SuperLUDistStructs::vec_int_t
{
  /// @brief vector
  std::vector<int_t> vec;
};

namespace
{
template <typename...>
constexpr bool dependent_false_v = false;
}

using namespace dolfinx;
using namespace dolfinx::la;

template <typename T>
void SuperLUDistSolver<T>::GridInfoDeleter::operator()(
    SuperLUDistStructs::gridinfo_t* gridinfo) const noexcept
{
  superlu_gridexit(gridinfo);
  delete gridinfo;
}

template <typename T>
void SuperLUDistSolver<T>::SuperMatrixDeleter::operator()(
    SuperLUDistStructs::SuperMatrix* supermatrix) const noexcept
{
  Destroy_SuperMatrix_Store_dist(supermatrix);
  delete supermatrix;
}

template <typename T>
void SuperLUDistSolver<T>::VecIntDeleter::operator()(
    SuperLUDistStructs::vec_int_t* vec) const noexcept
{
  delete vec;
}

template <typename T>
SuperLUDistSolver<T>::SuperLUDistSolver(
    std::shared_ptr<const MatrixCSR<T>> Amat, bool verbose)
    : _gridinfo(new SuperLUDistStructs::gridinfo_t, GridInfoDeleter{}),
      _supermatrix(new SuperLUDistStructs::SuperMatrix, SuperMatrixDeleter{}),
      _Amat(Amat), cols(new SuperLUDistStructs::vec_int_t, VecIntDeleter{}),
      _verbose(verbose)
{
  int size = dolfinx::MPI::size(Amat->comm());

  int nprow = size;
  int npcol = 1;

  spdlog::info("Start gridinit");
  superlu_gridinit(Amat->comm(), nprow, npcol, _gridinfo.get());
  spdlog::info("Finished gridinit");
  set_operator(*Amat);
}
//---------------------------------------------------------------------------------------
template <typename T>
void SuperLUDistSolver<T>::set_operator(const la::MatrixCSR<T>& Amat)
{
  spdlog::info("Start set_operator");
  // Global size
  int m = Amat.index_map(0)->size_global();
  int n = Amat.index_map(1)->size_global();
  if (m != n)
    throw std::runtime_error("Cannot solve non-square system");

  // Number of local rows
  int m_loc = Amat.num_owned_rows();

  // First row
  int first_row = Amat.index_map(0)->local_range()[0];

  // Local number of non-zeros
  int nnz_loc = Amat.row_ptr()[m_loc];
  cols->vec.resize(nnz_loc);
  rowptr.resize(m_loc + 1);

  // Copy row_ptr from int64
  std::copy(Amat.row_ptr().begin(),
            std::next(Amat.row_ptr().begin(), m_loc + 1), rowptr.begin());

  // Convert local to global indices (and cast to int_t)
  std::vector<std::int64_t> global_col_indices(
      Amat.index_map(1)->global_indices());
  std::transform(Amat.cols().begin(), std::next(Amat.cols().begin(), nnz_loc),
                 cols->vec.begin(), [&](std::int64_t local_index) -> int_t
                 { return global_col_indices[local_index]; });

  auto Amatdata = const_cast<T*>(Amat.values().data());
  if constexpr (std::is_same_v<T, double>)
  {
    dCreate_CompRowLoc_Matrix_dist(_supermatrix.get(), m, n, nnz_loc, m_loc,
                                   first_row, Amatdata, cols->vec.data(),
                                   rowptr.data(), SLU_NR_loc, SLU_D, SLU_GE);
  }
  else if constexpr (std::is_same_v<T, float>)
  {
    sCreate_CompRowLoc_Matrix_dist(_supermatrix.get(), m, n, nnz_loc, m_loc,
                                   first_row, Amatdata, cols->vec.data(),
                                   rowptr.data(), SLU_NR_loc, SLU_S, SLU_GE);
  }
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    zCreate_CompRowLoc_Matrix_dist(
        _supermatrix.get(), m, n, nnz_loc, m_loc, first_row,
        reinterpret_cast<doublecomplex*>(Amatdata), cols->vec.data(),
        rowptr.data(), SLU_NR_loc, SLU_Z, SLU_GE);
  }
  else
  {
    static_assert(dependent_false_v<T>, "Invalid scalar type");
  }
  spdlog::info("Finished set_operator");
}
//---------------------------------------------------------------------------------------
template <typename T>
int SuperLUDistSolver<T>::solve(const la::Vector<T>& bvec,
                                la::Vector<T>& uvec) const
{
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

  // Copy b to u (SuperLU_DIST reads b from u and then overwrites u with
  // solution)
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
    pdgssvx(&options, _supermatrix.get(), &ScalePermstruct, uvec.array().data(),
            ldb, nrhs, _gridinfo.get(), &LUstruct, &SOLVEstruct, berr.data(),
            &stat, &info);

    spdlog::info("Finalize solve");
    dSolveFinalize(&options, &SOLVEstruct);
    dScalePermstructFree(&ScalePermstruct);
    dLUstructFree(&LUstruct);
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
    psgssvx(&options, _supermatrix.get(), &ScalePermstruct, uvec.array().data(),
            ldb, nrhs, _gridinfo.get(), &LUstruct, &SOLVEstruct, berr.data(),
            &stat, &info);

    spdlog::info("Finalize solve");
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
    pzgssvx(&options, _supermatrix.get(), &ScalePermstruct,
            reinterpret_cast<doublecomplex*>(uvec.array().data()), ldb, nrhs,
            _gridinfo.get(), &LUstruct, &SOLVEstruct, berr.data(), &stat,
            &info);

    spdlog::info("Finalize solve");
    zSolveFinalize(&options, &SOLVEstruct);
    zScalePermstructFree(&ScalePermstruct);
    zLUstructFree(&LUstruct);
  }
  else
  {
    static_assert(dependent_false_v<T>, "Invalid scalar type");
  }
  spdlog::info("Finished solve");

  if (info != 0 and dolfinx::MPI::rank(_Amat->comm()) == 0)
  {
    std::cout << "SuperLU_DIST p*gssvx() error: " << info << std::endl
              << std::flush;
    spdlog::info("SuperLU_DIST p*gssvx() error: {}", info);
  }

  if (_verbose)
    PStatPrint(&options, &stat, _gridinfo.get());
  PStatFree(&stat);

  return info;
}
//---------------------------------------------------------------------------------------
template class la::SuperLUDistSolver<double>;
template class la::SuperLUDistSolver<float>;
template class la::SuperLUDistSolver<std::complex<double>>;

#endif

// Copyright (C) 2026 Jack S. Hale, Chris N. Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_SUPERLU_DIST

#include "superlu_dist.h"
extern "C"
{
#include <superlu_ddefs.h>
#include <superlu_sdefs.h>
#include <superlu_zdefs.h>
}
#include <algorithm>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::la;

// Trick for declaring anonymous typedef structs from SuperLU_DIST
struct dolfinx::la::SuperLUDistStructs::SuperMatrix : public ::SuperMatrix
{
};

/// Struct holding vector of type int_t
struct dolfinx::la::SuperLUDistStructs::vec_int_t
{
  /// @brief vector
  std::vector<int_t> vec;
};

void SuperMatrixDeleter::operator()(
    SuperLUDistStructs::SuperMatrix* supermatrix) const noexcept
{
  Destroy_SuperMatrix_Store_dist(supermatrix);
  delete supermatrix;
}

namespace
{
template <typename...>
constexpr bool dependent_false_v = false;

std::vector<int_t> col_indices(const auto& A)
{
  // Local number of non-zeros
  std::int32_t m_loc = A.num_owned_rows();
  std::int64_t nnz_loc = A.row_ptr().at(m_loc);

  std::vector global_indices(A.index_map(1)->global_indices());
  std::vector<int_t> col_indices(nnz_loc);
  std::transform(A.cols().begin(), std::next(A.cols().begin(), nnz_loc),
                 col_indices.begin(), [&global_indices](auto idx) -> int_t
                 { return global_indices[idx]; });
  return col_indices;
}

std::vector<int_t> row_indices(const auto& A)
{
  return std::vector<int_t>(
      A.row_ptr().begin(),
      std::next(A.row_ptr().begin(), A.num_owned_rows() + 1));
}

template <typename T>
std::unique_ptr<SuperLUDistStructs::SuperMatrix, SuperMatrixDeleter>
create_supermatrix(const auto& A, auto& rowptr, auto& cols)
{
  spdlog::info("Start set_operator");

  auto map0 = A.index_map(0);
  auto map1 = A.index_map(1);

  // Global size
  std::int64_t m = map0->size_global();
  std::int64_t n = map1->size_global();
  if (m != n)
    throw std::runtime_error("Cannot solve non-square system");

  // Number of local rows, first row and local number of non-zeros
  std::int32_t m_loc = A.num_owned_rows();
  std::int64_t first_row = map0->local_range().front();
  std::int64_t nnz_loc = A.row_ptr().at(m_loc);

  std::unique_ptr<SuperLUDistStructs::SuperMatrix, SuperMatrixDeleter> p(
      new SuperLUDistStructs::SuperMatrix, SuperMatrixDeleter{});

  // Note that the SuperMatrix shares the underlying data of A.
  T* Amatdata = const_cast<T*>(A.values().data());
  if constexpr (std::is_same_v<T, double>)
  {
    dCreate_CompRowLoc_Matrix_dist(p.get(), m, n, nnz_loc, m_loc, first_row,
                                   Amatdata, cols.vec.data(), rowptr.vec.data(),
                                   SLU_NR_loc, SLU_D, SLU_GE);
  }
  else if constexpr (std::is_same_v<T, float>)
  {
    sCreate_CompRowLoc_Matrix_dist(p.get(), m, n, nnz_loc, m_loc, first_row,
                                   Amatdata, cols.vec.data(), rowptr.vec.data(),
                                   SLU_NR_loc, SLU_S, SLU_GE);
  }
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    zCreate_CompRowLoc_Matrix_dist(p.get(), m, n, nnz_loc, m_loc, first_row,
                                   reinterpret_cast<doublecomplex*>(Amatdata),
                                   cols.vec.data(), rowptr.vec.data(),
                                   SLU_NR_loc, SLU_Z, SLU_GE);
  }
  else
    static_assert(dependent_false_v<T>, "Invalid scalar type");

  spdlog::info("Finished set_operator");
  return p;
}
} // namespace

//----------------------------------------------------------------------------
template <typename T>
SuperLUDistMatrix<T>::SuperLUDistMatrix(std::shared_ptr<const MatrixCSR<T>> A,
                                        bool verbose)
    : _Amat(A),
      _cols(std::make_unique<SuperLUDistStructs::vec_int_t>(col_indices(*A))),
      _rowptr(std::make_unique<SuperLUDistStructs::vec_int_t>(row_indices(*A))),
      _supermatrix(create_supermatrix<T>(*A, *_rowptr, *_cols)),
      _verbose(verbose)
{
}

//----------------------------------------------------------------------------
template <typename T>
const la::MatrixCSR<T>& SuperLUDistMatrix<T>::Amat() const
{
  assert(_Amat);
  return *_Amat;
}

//----------------------------------------------------------------------------
template <typename T>
SuperLUDistStructs::SuperMatrix* SuperLUDistMatrix<T>::supermatrix() const
{
  return _supermatrix.get();
}

//----------------------------------------------------------------------------
// Trick for declaring anonymous typedef structs from SuperLU_DIST
struct dolfinx::la::SuperLUDistStructs::gridinfo_t : public ::gridinfo_t
{
};

//----------------------------------------------------------------------------
void GridInfoDeleter::operator()(
    SuperLUDistStructs::gridinfo_t* gridinfo) const noexcept
{
  superlu_gridexit(gridinfo);
  delete gridinfo;
}

//----------------------------------------------------------------------------
template <typename T>
SuperLUDistSolver<T>::SuperLUDistSolver(std::shared_ptr<const MatrixCSR<T>> A,
                                        bool verbose)
    : _A_superlu_mat(SuperLUDistMatrix<T>(A, verbose)),
      _gridinfo(
          [comm = A->comm()]
          {
            int nprow = dolfinx::MPI::size(comm);
            int npcol = 1;
            std::unique_ptr<SuperLUDistStructs::gridinfo_t, GridInfoDeleter> p(
                new SuperLUDistStructs::gridinfo_t, GridInfoDeleter{});
            superlu_gridinit(comm, nprow, npcol, p.get());
            return p;
          }()),
      _verbose(verbose)
{
}

//----------------------------------------------------------------------------
template <typename T>
int SuperLUDistSolver<T>::solve(const la::Vector<T>& b, la::Vector<T>& u) const
{
  int_t m = _A_superlu_mat.supermatrix()->nrow;
  int_t m_loc = ((NRformat_loc*)(_A_superlu_mat.supermatrix()->Store))->m_loc;

  // RHS
  int_t ldb = m_loc;
  int_t nrhs = 1;

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
  std::copy_n(b.array().begin(), m_loc, u.array().begin());

  std::vector<scalar_value_t<T>> berr(nrhs);
  if constexpr (std::is_same_v<T, double>)
  {
    spdlog::info("Start solve [float64]");
    dScalePermstruct_t ScalePermstruct;
    dLUstruct_t LUstruct;
    dScalePermstructInit(m, m, &ScalePermstruct);
    dLUstructInit(m, &LUstruct);
    dSOLVEstruct_t SOLVEstruct;

    spdlog::info("Call SuperLU_DIST pdgssvx()");
    pdgssvx(&options, _A_superlu_mat.supermatrix(), &ScalePermstruct,
            u.array().data(), ldb, nrhs, _gridinfo.get(), &LUstruct,
            &SOLVEstruct, berr.data(), &stat, &info);

    spdlog::info("Finalize solve");
    dSolveFinalize(&options, &SOLVEstruct);
    dScalePermstructFree(&ScalePermstruct);
    dLUstructFree(&LUstruct);
  }
  else if constexpr (std::is_same_v<T, float>)
  {
    spdlog::info("Start solve [float32]");
    sScalePermstruct_t ScalePermstruct;
    sLUstruct_t LUstruct;
    sScalePermstructInit(m, m, &ScalePermstruct);
    sLUstructInit(m, &LUstruct);
    sSOLVEstruct_t SOLVEstruct;

    spdlog::info("Call SuperLU_DIST psgssvx()");
    psgssvx(&options, _A_superlu_mat.supermatrix(), &ScalePermstruct,
            u.array().data(), ldb, nrhs, _gridinfo.get(), &LUstruct,
            &SOLVEstruct, berr.data(), &stat, &info);

    spdlog::info("Finalize solve");
    sSolveFinalize(&options, &SOLVEstruct);
    sScalePermstructFree(&ScalePermstruct);
    sLUstructFree(&LUstruct);
  }
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    spdlog::info("Start solve [complex128]");
    zScalePermstruct_t ScalePermstruct;
    zLUstruct_t LUstruct;
    zScalePermstructInit(m, m, &ScalePermstruct);
    zLUstructInit(m, &LUstruct);
    zSOLVEstruct_t SOLVEstruct;

    spdlog::info("Call SuperLU_DIST pzgssvx()");
    pzgssvx(&options, _A_superlu_mat.supermatrix(), &ScalePermstruct,
            reinterpret_cast<doublecomplex*>(u.array().data()), ldb, nrhs,
            _gridinfo.get(), &LUstruct, &SOLVEstruct, berr.data(), &stat,
            &info);

    spdlog::info("Finalize solve");
    zSolveFinalize(&options, &SOLVEstruct);
    zScalePermstructFree(&ScalePermstruct);
    zLUstructFree(&LUstruct);
  }
  else
    static_assert(dependent_false_v<T>, "Invalid scalar type");
  spdlog::info("Finished solve");

  if (info != 0)
    spdlog::info("SuperLU_DIST p*gssvx() error: {}", info);

  if (_verbose)
    PStatPrint(&options, &stat, _gridinfo.get());
  PStatFree(&stat);

  return info;
}

//----------------------------------------------------------------------------
template class la::SuperLUDistSolver<double>;
template class la::SuperLUDistSolver<float>;
template class la::SuperLUDistSolver<std::complex<double>>;
//----------------------------------------------------------------------------
#endif

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
#include <dolfinx/common/Timer.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <stdexcept>
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

template <typename V, typename W>
void option_setter(W& option, const std::vector<V>& values,
                   const std::vector<std::string>& value_names,
                   const std::string value_in)
{
  for (std::size_t i = 0; i < value_names.size(); ++i)
  {
    if (value_in == value_names[i])
    {
      option = values[i];
      spdlog::info("Set to {}", value_in);
      return;
    }
  }
  std::runtime_error("Invalid option for SuperLU");
}

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
//----------------------------------------------------------------------------
std::vector<int_t> row_indices(const auto& A)
{
  return std::vector<int_t>(
      A.row_ptr().begin(),
      std::next(A.row_ptr().begin(), A.num_owned_rows() + 1));
}
//----------------------------------------------------------------------------
template <typename T>
std::unique_ptr<SuperLUDistStructs::SuperMatrix, SuperMatrixDeleter>
create_supermatrix(const auto& A, auto& rowptr, auto& cols)
{
  spdlog::info("Start create_supermatrix");

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

  // Check values fit into upper range of int_t.
  auto check = [](std::int64_t x)
  {
    if (x >= static_cast<std::int64_t>(std::numeric_limits<int_t>::max()))
    {
      throw std::out_of_range(
          "Value outside upper range of SuperLU_DIST int_t.");
    }
  };
  check(m);
  check(n);
  check(m_loc);
  check(first_row);
  check(nnz_loc);

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

  spdlog::info("Finished create_supermatrix");
  return p;
}
} // namespace
//----------------------------------------------------------------------------
template <typename T>
SuperLUDistMatrix<T>::SuperLUDistMatrix(std::shared_ptr<const MatrixCSR<T>> A,
                                        bool verbose)
    : _matA(std::move(A)),
      _cols(
          std::make_unique<SuperLUDistStructs::vec_int_t>(col_indices(*_matA))),
      _rowptr(
          std::make_unique<SuperLUDistStructs::vec_int_t>(row_indices(*_matA))),
      _supermatrix(create_supermatrix<T>(*_matA, *_rowptr, *_cols)),
      _verbose(verbose)
{
}
//----------------------------------------------------------------------------
template <typename T>
SuperLUDistStructs::SuperMatrix* SuperLUDistMatrix<T>::supermatrix() const
{
  return _supermatrix.get();
}
//----------------------------------------------------------------------------
template <typename T>
const MatrixCSR<T>& SuperLUDistMatrix<T>::matA() const
{
  return *_matA;
}
//----------------------------------------------------------------------------
template class la::SuperLUDistMatrix<double>;
template class la::SuperLUDistMatrix<float>;
template class la::SuperLUDistMatrix<std::complex<double>>;
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Trick for declaring anonymous typedef structs from SuperLU_DIST
struct dolfinx::la::SuperLUDistStructs::gridinfo_t : public ::gridinfo_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::superlu_dist_options_t : public ::superlu_dist_options_t
{
};
//----------------------------------------------------------------------------
void GridInfoDeleter::operator()(
    SuperLUDistStructs::gridinfo_t* gridinfo) const noexcept
{
  superlu_gridexit(gridinfo);
  delete gridinfo;
}

void SuperLUDistOptionsDeleter::operator()(
    SuperLUDistStructs::superlu_dist_options_t* opt) const noexcept
{
  delete opt;
}

//----------------------------------------------------------------------------
template <typename T>
SuperLUDistSolver<T>::SuperLUDistSolver(std::shared_ptr<const MatrixCSR<T>> A,
                                        bool verbose)
    : _superlu_matA(SuperLUDistMatrix<T>(A, verbose)),
      _options(new SuperLUDistStructs::superlu_dist_options_t,
               SuperLUDistOptionsDeleter{}),
      _gridinfo(
          [comm = _superlu_matA->matA().comm()]
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
  set_default_options_dist(_options.get());
  _options->DiagInv = YES;
  _options->ReplaceTinyPivot = YES;
  if (!_verbose)
    _options->PrintStat = NO;
}

template <typename T>
void SuperLUDistSolver<T>::set_option(std::string option, std::string value)
{
  spdlog::info("Set SuperLU_DIST option {} to {}", option, value);
  const std::map<std::string, yes_no_t&> map_bool
      = {{"Equil", _options->Equil},
         {"DiagInv", _options->DiagInv},
         {"SymmetricMode", _options->SymmetricMode},
         {"PivotGrowth", _options->PivotGrowth},
         {"ConditionNumber", _options->ConditionNumber},
         {"ReplaceTinyPivot", _options->ReplaceTinyPivot},
         {"SolveInitialized", _options->SolveInitialized},
         {"RefineInitialized", _options->RefineInitialized},
         {"PrintStat", _options->PrintStat},
         {"lookahead_etree", _options->lookahead_etree},
         {"SymPattern", _options->SymPattern},
         {"Use_TensorCore", _options->Use_TensorCore},
         {"Algo3d", _options->Algo3d}};

  // Search in map_bool first
  auto it = map_bool.find(option);
  if (it != map_bool.end())
  {
    if (value == "True")
    {
      spdlog::info("Set {} to True", option);
      it->second = YES;
    }
    else if (value == "False")
    {
      spdlog::info("Set {} to False", option);
      it->second = NO;
    }
    else
    {
      throw std::runtime_error("Cannot set option to value requested");
    }
  }

  // Search some enum types
  if (option == "Fact")
  {
    option_setter<fact_t>(
        _options->Fact,
        {DOFACT, SamePattern, SamePattern_SameRowPerm, FACTORED},
        {"DOFACT", "SamePattern", "SamePattern_SameRowPerm", "FACTORED"},
        value);
  }
  else if (option == "Trans")
  {
    option_setter<trans_t>(_options->Trans, {NOTRANS, TRANS, CONJ},
                           {"NOTRANS", "TRANS", "CONJ"}, value);
  }
  else if (option == "ColPerm")
  {
    option_setter<colperm_t>(
        _options->ColPerm,
        {NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD, METIS_AT_PLUS_A, PARMETIS,
         METIS_ATA, ZOLTAN, MY_PERMC},
        {"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD", "METIS_AT_PLUS_A",
         "PARMETIS", "METIS_ATA", "ZOLTAN", "MY_PERMC"},
        value);
  }
  else if (option == "RowPerm")
  {
    option_setter<rowperm_t>(
        _options->RowPerm,
        {NOROWPERM, LargeDiag_MC64, LargeDiag_HWPM, MY_PERMR},
        {"NOROWPERM", "LargeDiag_MC64", "LargeDiag_HWPM", "MY_PERMR"}, value);
  }
  else
  {
    std::runtime_error("Unsupported option");
  }
}
//----------------------------------------------------------------------------
template <typename T>
int SuperLUDistSolver<T>::solve(const la::Vector<T>& b, la::Vector<T>& u) const
{
  common::Timer tsolve("SuperLU Solve");
  int_t m = _superlu_matA.supermatrix()->nrow;
  int_t m_loc = ((NRformat_loc*)(_superlu_matA.supermatrix()->Store))->m_loc;

  // RHS
  int_t ldb = m_loc;
  int_t nrhs = 1;

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
    pdgssvx(&options, _superlu_matA->supermatrix(), &ScalePermstruct,
            u.array().data(), ldb, nrhs, _gridinfo.get(), &LUstruct,
            &SOLVEstruct, berr.data(), &stat, &info);

    spdlog::info("Finalize solve");
    dSolveFinalize(_options.get(), &SOLVEstruct);
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
    psgssvx(&options, _superlu_matA->supermatrix(), &ScalePermstruct,
            u.array().data(), ldb, nrhs, _gridinfo.get(), &LUstruct,
            &SOLVEstruct, berr.data(), &stat, &info);

    spdlog::info("Finalize solve");
    sSolveFinalize(_options.get(), &SOLVEstruct);
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
    pzgssvx(&options, _superlu_matA->supermatrix(), &ScalePermstruct,
            reinterpret_cast<doublecomplex*>(u.array().data()), ldb, nrhs,
            _gridinfo.get(), &LUstruct, &SOLVEstruct, berr.data(), &stat,
            &info);

    spdlog::info("Finalize solve");
    zSolveFinalize(_options.get(), &SOLVEstruct);
    zScalePermstructFree(&ScalePermstruct);
    zLUstructFree(&LUstruct);
  }
  else
    static_assert(dependent_false_v<T>, "Invalid scalar type");
  spdlog::info("Finished solve");

  if (info != 0)
    spdlog::info("SuperLU_DIST p*gssvx() error: {}", info);

  if (_verbose)
    PStatPrint(_options.get(), &stat, _gridinfo.get());
  PStatFree(&stat);

  return info;
}
//----------------------------------------------------------------------------
template class la::SuperLUDistSolver<double>;
template class la::SuperLUDistSolver<float>;
template class la::SuperLUDistSolver<std::complex<double>>;
//----------------------------------------------------------------------------
#endif

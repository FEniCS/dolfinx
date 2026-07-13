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
#include <array>
#include <cmath>
#include <dolfinx/common/Timer.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <initializer_list>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::la;

// Trick for declaring anonymous typedef structs from SuperLU_DIST
struct dolfinx::la::SuperLUDistStructs::SuperMatrix : public ::SuperMatrix
{
};
//----------------------------------------------------------------------------
/// Struct holding vector of type int_t
struct dolfinx::la::SuperLUDistStructs::vec_int_t
{
  /// @brief vector
  std::vector<int_t> vec;
};
//----------------------------------------------------------------------------
void SuperMatrixDeleter::operator()(
    SuperLUDistStructs::SuperMatrix* supermatrix) const noexcept
{
  Destroy_SuperMatrix_Store_dist(supermatrix);
  delete supermatrix;
}
//----------------------------------------------------------------------------
namespace
{
template <typename...>
constexpr bool always_false_v = false;

// Expand MatrixCSR block column indices to flattened column indices.
std::vector<int_t> col_indices(const auto& A)
{
  std::array<int, 2> bs = A.block_size();
  std::int32_t m_loc_block = A.num_owned_rows();
  std::int64_t nnz_loc_block = A.row_ptr().at(m_loc_block);
  std::vector global_indices(A.index_map(1)->global_indices());

  if (bs[0] == 1 and bs[1] == 1)
  {
    std::vector<int_t> col_indices(nnz_loc_block);
    std::transform(A.cols().begin(), std::next(A.cols().begin(), nnz_loc_block),
                   col_indices.begin(), [&global_indices](auto idx) -> int_t
                   { return global_indices[idx]; });
    return col_indices;
  }

  std::vector<int_t> col_indices(nnz_loc_block * bs[0] * bs[1]);
  const auto& A_cols = A.cols();
  const auto& A_rowptr = A.row_ptr();
  std::int64_t pos = 0;
  for (std::int32_t i = 0; i < m_loc_block; ++i)
  {
    for (int i0 = 0; i0 < bs[0]; ++i0)
    {
      for (std::int64_t j = A_rowptr[i]; j < A_rowptr[i + 1]; ++j)
      {
        int_t col_block = global_indices[A_cols[j]];
        for (int i1 = 0; i1 < bs[1]; ++i1)
          col_indices[pos++] = col_block * bs[1] + i1;
      }
    }
  }
  return col_indices;
}
//----------------------------------------------------------------------------
// Expand MatrixCSR block row pointer to flattened row pointer.
std::vector<int_t> row_indices(const auto& A)
{
  std::array<int, 2> bs = A.block_size();
  std::int32_t m_loc_block = A.num_owned_rows();
  const auto& A_rowptr = A.row_ptr();

  if (bs[0] == 1 and bs[1] == 1)
  {
    return std::vector<int_t>(A_rowptr.begin(),
                              std::next(A_rowptr.begin(), m_loc_block + 1));
  }

  // Write the per-scalar-row entry counts into `flattened_rowptr[1:]`, with
  // each block-row contributing `bs[0]` copies.
  std::vector<int_t> flattened_rowptr(m_loc_block * bs[0] + 1);
  flattened_rowptr[0] = A_rowptr[0] * bs[1];
  for (std::int32_t i = 0; i < m_loc_block; ++i)
  {
    int_t delta = (A_rowptr[i + 1] - A_rowptr[i]) * bs[1];
    std::fill_n(std::next(flattened_rowptr.begin(), 1 + i * bs[0]), bs[0],
                delta);
  }
  std::inclusive_scan(std::next(flattened_rowptr.begin()),
                      flattened_rowptr.end(),
                      std::next(flattened_rowptr.begin()));
  return flattened_rowptr;
}
//----------------------------------------------------------------------------
// Expand MatrixCSR block values to flattened CSR layout.
template <typename T>
std::vector<T> matrix_values(const MatrixCSR<T>& A)
{
  std::array<int, 2> bs = A.block_size();
  std::int32_t m_loc_block = A.num_owned_rows();
  std::int64_t nnz_loc_block = A.row_ptr().at(m_loc_block);

  if (bs[0] == 1 and bs[1] == 1)
  {
    return std::vector<T>(A.values().begin(),
                          std::next(A.values().begin(), nnz_loc_block));
  }

  std::vector<T> flattened_values(nnz_loc_block * bs[0] * bs[1]);
  const auto& A_values = A.values();
  const auto& A_rowptr = A.row_ptr();
  std::int64_t pos = 0;
  for (std::int32_t i = 0; i < m_loc_block; ++i)
  {
    for (int i0 = 0; i0 < bs[0]; ++i0)
    {
      for (std::int64_t j = A_rowptr[i]; j < A_rowptr[i + 1]; ++j)
      {
        for (int i1 = 0; i1 < bs[1]; ++i1)
          flattened_values[pos++]
              = A_values[j * bs[0] * bs[1] + i0 * bs[1] + i1];
      }
    }
  }
  return flattened_values;
}
//----------------------------------------------------------------------------
template <typename T>
std::unique_ptr<SuperLUDistStructs::SuperMatrix, SuperMatrixDeleter>
create_supermatrix(const auto& A, auto& A_mat_values, auto& rowptr, auto& cols)
{
  spdlog::info("Start create_supermatrix");

  auto map0 = A.index_map(0);
  auto map1 = A.index_map(1);
  std::array<int, 2> bs = A.block_size();

  // Global size (scalar, after block expansion)
  std::int64_t m = map0->size_global() * bs[0];
  std::int64_t n = map1->size_global() * bs[1];
  if (m != n)
    throw std::runtime_error("Cannot solve non-square system");

  // Number of local rows, first row and local number of non-zeros.
  std::int32_t m_loc = A.num_owned_rows() * bs[0];
  std::int64_t first_row = map0->local_range().front() * bs[0];
  std::int64_t nnz_loc = A.row_ptr().at(A.num_owned_rows()) * bs[0] * bs[1];

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

  if constexpr (std::is_same_v<T, double>)
  {
    dCreate_CompRowLoc_Matrix_dist(
        p.get(), m, n, nnz_loc, m_loc, first_row, A_mat_values.data(),
        cols.vec.data(), rowptr.vec.data(), SLU_NR_loc, SLU_D, SLU_GE);
  }
  else if constexpr (std::is_same_v<T, float>)
  {
    sCreate_CompRowLoc_Matrix_dist(
        p.get(), m, n, nnz_loc, m_loc, first_row, A_mat_values.data(),
        cols.vec.data(), rowptr.vec.data(), SLU_NR_loc, SLU_S, SLU_GE);
  }
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    zCreate_CompRowLoc_Matrix_dist(
        p.get(), m, n, nnz_loc, m_loc, first_row,
        reinterpret_cast<doublecomplex*>(A_mat_values.data()), cols.vec.data(),
        rowptr.vec.data(), SLU_NR_loc, SLU_Z, SLU_GE);
  }
  else
    static_assert(always_false_v<T>, "Invalid scalar type");

  spdlog::info("Finished create_supermatrix");
  return p;
}
} // namespace
//----------------------------------------------------------------------------
template <typename T>
SuperLUDistMatrix<T>::SuperLUDistMatrix(const MatrixCSR<T>& A)
    : _comm(A.comm()), _matA_values(matrix_values(A)),
      _cols(std::make_unique<SuperLUDistStructs::vec_int_t>(col_indices(A))),
      _rowptr(std::make_unique<SuperLUDistStructs::vec_int_t>(row_indices(A))),
      _supermatrix(create_supermatrix<T>(A, _matA_values, *_rowptr, *_cols))
{
}
/// @brief Get MPI communicator that matrix is defined on.
template <typename T>
MPI_Comm SuperLUDistMatrix<T>::comm() const
{
  return _comm.comm();
}
//----------------------------------------------------------------------------
template <typename T>
SuperLUDistStructs::SuperMatrix* SuperLUDistMatrix<T>::supermatrix() const
{
  return _supermatrix.get();
}
//----------------------------------------------------------------------------
template class la::SuperLUDistMatrix<double>;
template class la::SuperLUDistMatrix<float>;
template class la::SuperLUDistMatrix<std::complex<double>>;
//----------------------------------------------------------------------------

namespace
{
template <typename V, typename W>
void option_setter(W& option, std::string_view value_in,
                   std::string_view option_name,
                   std::initializer_list<V> values,
                   std::initializer_list<std::string_view> value_names)
{
  // TODO: Can be done nicely with std::views::zip in C++23.
  for (auto i : std::views::iota(std::size_t{0}, value_names.size()))
  {
    if (value_in == *(value_names.begin() + i))
    {
      option = *(values.begin() + i);
      spdlog::info("Set {} to {}", option_name, value_in);
      return;
    }
  }
  throw std::runtime_error("Unsupported value");
}
} // namespace

//----------------------------------------------------------------------------
// Trick for declaring anonymous typedef structs from SuperLU_DIST
struct dolfinx::la::SuperLUDistStructs::gridinfo_t : public ::gridinfo_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::superlu_dist_options_t
    : public ::superlu_dist_options_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::dScalePermstruct_t
    : public ::dScalePermstruct_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::sScalePermstruct_t
    : public ::sScalePermstruct_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::zScalePermstruct_t
    : public ::zScalePermstruct_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::dLUstruct_t : public ::dLUstruct_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::sLUstruct_t : public ::sLUstruct_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::zLUstruct_t : public ::zLUstruct_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::dSOLVEstruct_t : public ::dSOLVEstruct_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::sSOLVEstruct_t : public ::sSOLVEstruct_t
{
};
//----------------------------------------------------------------------------
struct dolfinx::la::SuperLUDistStructs::zSOLVEstruct_t : public ::zSOLVEstruct_t
{
};
//----------------------------------------------------------------------------
void GridInfoDeleter::operator()(
    SuperLUDistStructs::gridinfo_t* g) const noexcept
{
  superlu_gridexit(g);
  delete g;
};
//----------------------------------------------------------------------------
void ScalePermStructDeleter::operator()(
    SuperLUDistStructs::dScalePermstruct_t* s) const noexcept
{
  dScalePermstructFree(s);
  delete s;
};
//----------------------------------------------------------------------------
void ScalePermStructDeleter::operator()(
    SuperLUDistStructs::sScalePermstruct_t* s) const noexcept
{
  sScalePermstructFree(s);
  delete s;
};
//----------------------------------------------------------------------------
void ScalePermStructDeleter::operator()(
    SuperLUDistStructs::zScalePermstruct_t* s) const noexcept
{
  zScalePermstructFree(s);
  delete s;
};
//----------------------------------------------------------------------------
void LUStructDeleter::operator()(
    SuperLUDistStructs::dLUstruct_t* l) const noexcept
{
  dLUstructFree(l);
  delete l;
};
//----------------------------------------------------------------------------
void LUStructDeleter::operator()(
    SuperLUDistStructs::sLUstruct_t* l) const noexcept
{
  sLUstructFree(l);
  delete l;
};
//----------------------------------------------------------------------------
void LUStructDeleter::operator()(
    SuperLUDistStructs::zLUstruct_t* l) const noexcept
{
  zLUstructFree(l);
  delete l;
};
//----------------------------------------------------------------------------
void SolveStructDeleter::operator()(
    SuperLUDistStructs::dSOLVEstruct_t* s) const noexcept
{
  if (o->SolveInitialized)
    dSolveFinalize(o, s);
  delete s;
};
//----------------------------------------------------------------------------
void SolveStructDeleter::operator()(
    SuperLUDistStructs::sSOLVEstruct_t* s) const noexcept
{
  if (o->SolveInitialized)
    sSolveFinalize(o, s);
  delete s;
};
//----------------------------------------------------------------------------
void SolveStructDeleter::operator()(
    SuperLUDistStructs::zSOLVEstruct_t* s) const noexcept
{
  if (o->SolveInitialized)
    zSolveFinalize(o, s);
  delete s;
};
//----------------------------------------------------------------------------
template <typename T>
SuperLUDistSolver<T>::SuperLUDistSolver(
    std::shared_ptr<const SuperLUDistMatrix<T>> A)
    : _superlu_matA(std::move(A)),
      _options(
          []
          {
            auto o = std::make_unique<
                SuperLUDistStructs::superlu_dist_options_t>();
            set_default_options_dist(o.get());
            o->PrintStat = NO;
            return o;
          }()),
      _gridinfo(
          [comm = _superlu_matA->comm()]
          {
            // Computes a balanced 2D process grid - see PETSc interface
            int size = dolfinx::MPI::size(comm);
            int nprow = static_cast<int>(std::sqrt(static_cast<double>(size)));
            while (size % nprow != 0)
              --nprow;
            int npcol = size / nprow;
            std::unique_ptr<SuperLUDistStructs::gridinfo_t, GridInfoDeleter> p(
                new SuperLUDistStructs::gridinfo_t, GridInfoDeleter{});
            superlu_gridinit(comm, nprow, npcol, p.get());
            return p;
          }()),
      _scalepermstruct(
          [m = _superlu_matA->supermatrix()->nrow]
          {
            std::unique_ptr<typename map_t<T>::ScalePermstruct_t,
                            ScalePermStructDeleter>
                s(new map_t<T>::ScalePermstruct_t, ScalePermStructDeleter{});

            if constexpr (std::is_same_v<T, double>)
            {
              dScalePermstructInit(m, m, s.get());
            }
            else if constexpr (std::is_same_v<T, float>)
            {
              sScalePermstructInit(m, m, s.get());
            }
            else if constexpr (std::is_same_v<T, std::complex<double>>)
            {
              zScalePermstructInit(m, m, s.get());
            }
            else
              static_assert(always_false_v<T>, "Invalid scalar type");
            return s;
          }()),
      _lustruct(
          [m = _superlu_matA->supermatrix()->nrow]
          {
            std::unique_ptr<typename map_t<T>::LUstruct_t, LUStructDeleter> l(
                new map_t<T>::LUstruct_t, LUStructDeleter{});

            if constexpr (std::is_same_v<T, double>)
            {
              dLUstructInit(m, l.get());
            }
            else if constexpr (std::is_same_v<T, float>)
            {
              sLUstructInit(m, l.get());
            }
            else if constexpr (std::is_same_v<T, std::complex<double>>)
            {
              zLUstructInit(m, l.get());
            }
            else
              static_assert(always_false_v<T>, "Invalid scalar type");
            return l;
          }()),
      _solvestruct(new typename map_t<T>::SOLVEstruct_t{},
                   SolveStructDeleter{_options.get()})
{
}
//----------------------------------------------------------------------------
template <typename T>
void SuperLUDistSolver<T>::set_options(
    SuperLUDistStructs::superlu_dist_options_t options)
{
  *_options = options;
}
//----------------------------------------------------------------------------
template <typename T>
void SuperLUDistSolver<T>::set_option(std::string name, std::string value)
{
  spdlog::info("Attempting to set option {} to {}", name, value);
  const std::map<std::string, std::reference_wrapper<yes_no_t>> map_bool
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
  auto it = map_bool.find(name);
  if (it != map_bool.end())
  {
    if (value == "YES")
    {
      spdlog::info("Set {} to YES", name);
      it->second.get() = YES;
      return;
    }
    else if (value == "NO")
    {
      spdlog::info("Set {} to NO", name);
      it->second.get() = NO;
      return;
    }
    else
    {
      throw std::runtime_error(
          "'Boolean' option values must be string 'YES' or 'NO'");
    }
  }

  // Search some enum types
  if (name == "Fact")
  {
    option_setter(
        _options->Fact, value, "Fact",
        {DOFACT, SamePattern, SamePattern_SameRowPerm, FACTORED},
        {"DOFACT", "SamePattern", "SamePattern_SameRowPerm", "FACTORED"});
  }
  else if (name == "Trans")
  {
    option_setter(_options->Trans, value, "Trans", {NOTRANS, TRANS, CONJ},
                  {"NOTRANS", "TRANS", "CONJ"});
  }
  else if (name == "ColPerm")
  {
    option_setter(_options->ColPerm, value, "ColPerm",
                  {NATURAL, MMD_ATA, MMD_AT_PLUS_A, COLAMD, METIS_AT_PLUS_A,
                   PARMETIS, METIS_ATA, ZOLTAN, MY_PERMC},
                  {"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD",
                   "METIS_AT_PLUS_A", "PARMETIS", "METIS_ATA", "ZOLTAN",
                   "MY_PERMC"});
  }
  else if (name == "RowPerm")
  {
    option_setter(
        _options->RowPerm, value, "RowPerm",
        {NOROWPERM, LargeDiag_MC64, LargeDiag_HWPM, MY_PERMR},
        {"NOROWPERM", "LargeDiag_MC64", "LargeDiag_HWPM", "MY_PERMR"});
  }
  else
  {
    throw std::runtime_error("Unsupported option name");
  }
}
//----------------------------------------------------------------------------
template <typename T>
SuperLUDistSolver<T>::~SuperLUDistSolver()
{
  if (_factored)
  {
    int_t n = _superlu_matA->supermatrix()->ncol;
    if constexpr (std::is_same_v<T, double>)
      dDestroy_LU(n, _gridinfo.get(), _lustruct.get());
    else if constexpr (std::is_same_v<T, float>)
      sDestroy_LU(n, _gridinfo.get(), _lustruct.get());
    else if constexpr (std::is_same_v<T, std::complex<double>>)
      zDestroy_LU(n, _gridinfo.get(), _lustruct.get());
    else
      static_assert(always_false_v<T>, "Invalid scalar type");
  }
}
//----------------------------------------------------------------------------
template <typename T>
void SuperLUDistSolver<T>::set_A(std::shared_ptr<const SuperLUDistMatrix<T>> A,
                                 std::string fact)
{
  if (A->supermatrix()->nrow != _superlu_matA->supermatrix()->nrow
      or A->supermatrix()->ncol != _superlu_matA->supermatrix()->ncol)
  {
    throw std::runtime_error(
        "New matrix A has different size to the matrix used to construct the "
        "solver.");
  }
  _superlu_matA = A;

  // See pddistribute in SuperLU_DIST: on Fact=DOFACT or Fact=SamePattern
  // pdgssvx overwrites LUstruct's internal pointers without freeing the
  // previous arrays, so a prior factorisation must be released first.
  // Fact=SamePattern_SameRowPerm reuses the existing arrays in place.
  if (fact == "DOFACT" or fact == "SamePattern")
  {
    if (_factored)
    {
      int_t n = _superlu_matA->supermatrix()->ncol;
      if constexpr (std::is_same_v<T, double>)
        dDestroy_LU(n, _gridinfo.get(), _lustruct.get());
      else if constexpr (std::is_same_v<T, float>)
        sDestroy_LU(n, _gridinfo.get(), _lustruct.get());
      else if constexpr (std::is_same_v<T, std::complex<double>>)
        zDestroy_LU(n, _gridinfo.get(), _lustruct.get());
      else
        static_assert(always_false_v<T>, "Invalid scalar type");
      _factored = false;
    }
    _options->Fact = (fact == "DOFACT") ? DOFACT : SamePattern;
  }
  else if (fact == "SamePattern_SameRowPerm")
  {
    _options->Fact = SamePattern_SameRowPerm;
  }
  else
  {
    throw std::runtime_error("set_A fact must be one of 'DOFACT', "
                             "'SamePattern', 'SamePattern_SameRowPerm'");
  }
}
//----------------------------------------------------------------------------
template <typename T>
int SuperLUDistSolver<T>::solve(const la::Vector<T>& b, la::Vector<T>& u)
{
  common::Timer tsolve("SuperLU_DIST solve");

  if (b.array().size() != u.array().size())
    throw std::runtime_error("b and u have incompatible size/layout");

  // Warn if Factor is DOFACT but solve already appears to be initialised.
  if (_options->Fact == DOFACT && _options->SolveInitialized == YES)
    spdlog::warn("Extra call to solve with option Fact set to DOFACT. "
                 "This leads to incorrect results; try FACTORED.");

  int_t m_loc = ((NRformat_loc*)(_superlu_matA->supermatrix()->Store))->m_loc;
  // RHS
  int_t ldb = m_loc;
  // TODO: Support for multiple right-hand sides?
  int_t nrhs = 1;

  int info = 0;
  SuperLUStat_t stat;
  PStatInit(&stat);

  // Copy b to u (SuperLU_DIST reads b from u and then overwrites u with
  // solution)
  std::copy_n(b.array().begin(), b.array().size(), u.array().begin());

  std::vector<scalar_value_t<T>> berr(nrhs);
  if constexpr (std::is_same_v<T, double>)
  {
    spdlog::info("Start solve [float64]");

    spdlog::info("Call SuperLU_DIST pdgssvx()");
    pdgssvx(_options.get(), _superlu_matA->supermatrix(),
            _scalepermstruct.get(), u.array().data(), ldb, nrhs,
            _gridinfo.get(), _lustruct.get(), _solvestruct.get(), berr.data(),
            &stat, &info);
  }
  else if constexpr (std::is_same_v<T, float>)
  {
    spdlog::info("Start solve [float32]");

    spdlog::info("Call SuperLU_DIST psgssvx()");
    psgssvx(_options.get(), _superlu_matA->supermatrix(),
            _scalepermstruct.get(), u.array().data(), ldb, nrhs,
            _gridinfo.get(), _lustruct.get(), _solvestruct.get(), berr.data(),
            &stat, &info);
  }
  else if constexpr (std::is_same_v<T, std::complex<double>>)
  {
    spdlog::info("Start solve [complex128]");

    spdlog::info("Call SuperLU_DIST pzgssvx()");
    pzgssvx(_options.get(), _superlu_matA->supermatrix(),
            _scalepermstruct.get(),
            reinterpret_cast<doublecomplex*>(u.array().data()), ldb, nrhs,
            _gridinfo.get(), _lustruct.get(), _solvestruct.get(), berr.data(),
            &stat, &info);
  }
  else
    static_assert(always_false_v<T>, "Invalid scalar type");

  if (info != 0)
    spdlog::info("SuperLU_DIST p*gssvx() error: {}", info);

  // pdgssvx allocates LUstruct internals during factorisation. Record this
  // so the destructor / set_A can call Destroy_LU to release them.
  _factored = true;

  PStatPrint(_options.get(), &stat, _gridinfo.get());
  PStatFree(&stat);

  spdlog::info("Finished solve");

  return info;
}
//----------------------------------------------------------------------------
template class la::SuperLUDistSolver<double>;
template class la::SuperLUDistSolver<float>;
template class la::SuperLUDistSolver<std::complex<double>>;
//----------------------------------------------------------------------------
#endif

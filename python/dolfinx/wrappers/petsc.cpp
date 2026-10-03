// Copyright (C) 2017-2026 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#if defined(HAS_PETSC) && defined(HAS_PETSC4PY)

#include "dolfinx_wrappers/petsc.h"
#include "dolfinx_wrappers/array.h"
#include "dolfinx_wrappers/pycoeff.h"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/petsc.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/petsc.h>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <petsc4py/petsc4py.h>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{
namespace nb = nanobind;

/// @brief Convert a list of (IndexMap pointer-like, block size) pairs
/// into the reference_wrapper form expected by dolfinx::la::petsc and
/// dolfinx::fem::petsc functions.
template <typename U>
std::vector<
    std::pair<std::reference_wrapper<const dolfinx::common::IndexMap>, int>>
to_index_map_refs(const std::vector<std::pair<U, int>>& maps)
{
  std::vector<
      std::pair<std::reference_wrapper<const dolfinx::common::IndexMap>, int>>
      _maps;
  std::ranges::transform(maps, std::back_inserter(_maps), [](auto& m)
                         { return std::pair{std::cref(*m.first), m.second}; });
  return _maps;
}

/// @brief Test if A has row and column block size 1, in which case
/// blocked and non-blocked insertion of dof indices are equivalent.
bool unit_block_size(Mat A)
{
  PetscInt bs0 = -1, bs1 = -1;
  dolfinx::common::petsc::check(MatGetBlockSizes(A, &bs0, &bs1),
                                "MatGetBlockSizes");
  return bs0 == 1 and bs1 == 1;
}

/// @brief Convert a petsc4py InsertMode, passed as a Python int, to a
/// PETSc InsertMode for matrix insertion.
InsertMode insert_mode(int mode)
{
  InsertMode _mode = static_cast<InsertMode>(mode);
  if (_mode != INSERT_VALUES and _mode != ADD_VALUES)
  {
    throw std::invalid_argument(
        "InsertMode must be INSERT_VALUES or ADD_VALUES.");
  }
  return _mode;
}

void petsc_la_module(nb::module_& m)
{
  if (import_petsc4py() != 0)
    throw std::runtime_error("Could not import petsc4py.");

  m.def(
      "create_matrix",
      [](dolfinx_wrappers::MPICommWrapper comm,
         const dolfinx::la::SparsityPattern& p,
         std::optional<std::string> type) -> Mat
      { return dolfinx::la::petsc::create_matrix(comm.get(), p, type); },
      nb::rv_policy::take_ownership, nb::arg("comm"), nb::arg("p"),
      nb::arg("type").none(), "Create a PETSc Mat from sparsity pattern.");

  m.def(
      "create_index_sets",
      [](const std::vector<std::pair<const dolfinx::common::IndexMap*, int>>&
             maps) -> std::vector<IS>
      {
        auto _maps = to_index_map_refs(maps);
        return dolfinx::la::petsc::create_index_sets(_maps);
      },
      nb::rv_policy::take_ownership, nb::arg("maps"));

  m.def(
      "scatter_local_vectors",
      [](Vec x,
         const std::vector<
             nb::ndarray<const PetscScalar, nb::ndim<1>, nb::c_contig>>& x_b,
         const std::vector<std::pair<
             std::shared_ptr<const dolfinx::common::IndexMap>, int>>& maps)
      {
        std::vector<std::span<const PetscScalar>> _x_b
            = dolfinx_wrappers::vec_of_spans(x_b);
        auto _maps = to_index_map_refs(maps);
        dolfinx::la::petsc::scatter_local_vectors(x, _x_b, _maps);
      },
      nb::arg("x"), nb::arg("x_b"), nb::arg("maps"),
      "Scatter the (ordered) list of sub vectors into a block "
      "vector.");

  m.def(
      "get_local_vectors",
      [](const Vec x,
         const std::vector<std::pair<
             std::shared_ptr<const dolfinx::common::IndexMap>, int>>& maps)
      {
        auto _maps = to_index_map_refs(maps);
        std::vector<std::vector<PetscScalar>> vecs
            = dolfinx::la::petsc::get_local_vectors(x, _maps);
        std::vector<nb::ndarray<PetscScalar, nb::numpy>> ret;
        std::ranges::transform(
            vecs, std::back_inserter(ret),
            [](auto& v) { return dolfinx_wrappers::as_nbarray(std::move(v)); });
        return ret;
      },
      nb::arg("x"), nb::arg("maps"),
      "Gather an (ordered) list of sub vectors from a block vector.");
  m.def(
      "set_diagonal",
      [](Mat A, nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> rows,
         nb::ndarray<const PetscScalar, nb::ndim<1>, nb::c_contig> diagonals,
         int mode)
      {
        dolfinx::la::set_diagonal(
            dolfinx::la::petsc::Matrix::set_fn(A, insert_mode(mode)),
            std::span(rows.data(), rows.size()),
            std::span<const PetscScalar>(diagonals.data(), diagonals.size()));
      },
      nb::arg("A"), nb::arg("rows"), nb::arg("diagonals"), nb::arg("mode"));
  m.def(
      "set_diagonal",
      [](Mat A, nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> rows,
         PetscScalar diagonal, int mode)
      {
        dolfinx::la::set_diagonal(
            dolfinx::la::petsc::Matrix::set_fn(A, insert_mode(mode)),
            std::span(rows.data(), rows.size()), diagonal);
      },
      nb::arg("A"), nb::arg("rows"), nb::arg("diagonal"), nb::arg("mode"));
}

void petsc_fem_module(nb::module_& m)
{
  dolfinx_wrappers::declare_petsc_discrete_operators<PetscScalar, PetscReal>(m);

  // Create PETSc vectors and matrices
  m.def(
      "create_vector_block",
      [](const std::vector<std::pair<
             std::shared_ptr<const dolfinx::common::IndexMap>, int>>& maps)
      {
        auto _maps = to_index_map_refs(maps);
        return dolfinx::fem::petsc::create_vector_block(_maps);
      },
      nb::rv_policy::take_ownership, nb::arg("maps"),
      "Create a monolithic vector for multiple (stacked) linear forms.");
  m.def(
      "create_vector_nest",
      [](const std::vector<std::pair<
             std::shared_ptr<const dolfinx::common::IndexMap>, int>>& maps)
      {
        auto _maps = to_index_map_refs(maps);
        return dolfinx::fem::petsc::create_vector_nest(_maps);
      },
      nb::rv_policy::take_ownership, nb::arg("maps"),
      "Create nested vector for multiple (stacked) linear forms.");
  m.def("create_matrix", &dolfinx::fem::petsc::create_matrix<PetscReal>,
        nb::rv_policy::take_ownership, nb::arg("a"), nb::arg("type").none(),
        "Create a PETSc Mat for bilinear form.");
  m.def("create_matrix_block",
        &dolfinx::fem::petsc::create_matrix_block<PetscReal>,
        nb::rv_policy::take_ownership, nb::arg("a"), nb::arg("type").none(),
        "Create monolithic sparse matrix for stacked bilinear forms.");
  m.def("create_matrix_nest",
        &dolfinx::fem::petsc::create_matrix_nest<PetscReal>,
        nb::rv_policy::take_ownership, nb::arg("a"), nb::arg("types").none(),
        "Create nested sparse matrix for bilinear forms.");

  // PETSc Matrices
  m.def(
      "assemble_matrix",
      [](Mat A, const dolfinx::fem::Form<PetscScalar, PetscReal>& a,
         nb::ndarray<const PetscScalar, nb::ndim<1>, nb::c_contig> constants,
         const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                        nb::ndarray<const PetscScalar, nb::ndim<2>,
                                    nb::c_contig>>& coefficients,
         nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig> dof_marker0,
         nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig> dof_marker1,
         bool unrolled)
      {
        std::span<const std::int8_t> _dof_marker0(dof_marker0.data(),
                                                  dof_marker0.size());
        std::span<const std::int8_t> _dof_marker1(dof_marker1.data(),
                                                  dof_marker1.size());
        if (unrolled)
        {
          auto set_fn = dolfinx::la::petsc::Matrix::set_block_expand_fn(
              A, a.function_spaces()[0]->dofmap()->bs(),
              a.function_spaces()[1]->dofmap()->bs(), ADD_VALUES);
          dolfinx::fem::assemble_matrix(
              set_fn, a, std::span(constants.data(), constants.size()),
              dolfinx_wrappers::py_to_cpp_coeffs(coefficients), _dof_marker0,
              _dof_marker1);
        }
        else
        {
          // Non-blocked insertion is cheaper than the blocked interface,
          // and equivalent when A has block size 1
          if (unit_block_size(A))
          {
            dolfinx::fem::assemble_matrix(
                dolfinx::la::petsc::Matrix::set_fn(A, ADD_VALUES), a,
                std::span(constants.data(), constants.size()),
                dolfinx_wrappers::py_to_cpp_coeffs(coefficients), _dof_marker0,
                _dof_marker1);
          }
          else
          {
            dolfinx::fem::assemble_matrix(
                dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES), a,
                std::span(constants.data(), constants.size()),
                dolfinx_wrappers::py_to_cpp_coeffs(coefficients), _dof_marker0,
                _dof_marker1);
          }
        }
      },
      nb::arg("A"), nb::arg("a"), nb::arg("constants"), nb::arg("coeffs"),
      nb::arg("dof_marker0"), nb::arg("dof_marker1"), nb::arg("unrolled"),
      "Assemble bilinear form into an existing PETSc matrix");
}

} // namespace

namespace dolfinx_wrappers
{
void petsc(nb::module_& m_fem, nb::module_& m_la)
{
  nb::module_ petsc_fem_mod
      = m_fem.def_submodule("petsc", "PETSc-specific finite element module");
  petsc_fem_module(petsc_fem_mod);

  nb::module_ petsc_la_mod
      = m_la.def_submodule("petsc", "PETSc-specific linear algebra module");
  petsc_la_module(petsc_la_mod);
}
} // namespace dolfinx_wrappers
#endif

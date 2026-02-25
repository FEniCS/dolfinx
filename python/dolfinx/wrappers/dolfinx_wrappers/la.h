// Copyright (C) 2017-2025 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "array.h"
#include "numpy_dtype.h"
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/la/utils.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <string>
#include <vector>

#if defined(HAS_SUPERLU_DIST)
#include <dolfinx/la/superlu_dist.h>
#endif

namespace dolfinx_wrappers
{

// InsertMode types for Python bindings
enum class PyInsertMode : std::uint8_t
{
  add,
  insert
};

/// Declare linear algebra objects (Vector, MatrixCSR) for a given scalar type
/// @param m The nanobind module
/// @param type String representation of the scalar type (e.g., "float64",
/// "complex128")
template <typename T>
void declare_la_objects(nanobind::module_& m, const std::string& type)
{
  namespace nb = nanobind;
  using namespace nb::literals;

  // dolfinx::la::Vector
  std::string pyclass_vector_name = std::string("Vector_") + type;
  nb::class_<dolfinx::la::Vector<T>>(m, pyclass_vector_name.c_str())
      .def(nb::init<std::shared_ptr<const dolfinx::common::IndexMap>, int>(),
           nb::arg("map"), nb::arg("bs"))
      .def(nb::init<const dolfinx::la::Vector<T>&>(), nb::arg("vec"))
      .def_prop_ro("dtype", [](const dolfinx::la::Vector<T>&)
                   { return dolfinx_wrappers::numpy_dtype_v<T>; })
      .def_prop_ro("index_map", &dolfinx::la::Vector<T>::index_map)
      .def_prop_ro("bs", &dolfinx::la::Vector<T>::bs)
      .def_prop_ro(
          "array",
          [](dolfinx::la::Vector<T>& self)
          {
            return nb::ndarray<T, nb::numpy>(self.array().data(),
                                             {self.array().size()});
          },
          nb::rv_policy::reference_internal)
      .def("scatter_forward",
           [](dolfinx::la::Vector<T>& self) { self.scatter_fwd(); })
      .def(
          "scatter_reverse",
          [](dolfinx::la::Vector<T>& self, PyInsertMode mode)
          {
            switch (mode)
            {
            case PyInsertMode::add: // Add
              self.scatter_rev(std::plus<T>());
              break;
            case PyInsertMode::insert: // Insert
              self.scatter_rev([](T /*a*/, T b) { return b; });
              break;
            default:
              throw std::runtime_error("InsertMode not recognized.");
              break;
            }
          },
          nb::arg("mode"));

  // dolfinx::la::MatrixCSR
  std::string pyclass_matrix_name = std::string("MatrixCSR_") + type;
  nb::class_<dolfinx::la::MatrixCSR<
      T, std::vector<T>, std::vector<std::int32_t>, std::vector<std::int64_t>>>(
      m, pyclass_matrix_name.c_str())
      .def(nb::init<const dolfinx::la::SparsityPattern&,
                    dolfinx::la::BlockMode>(),
           nb::arg("p"),
           nb::arg("block_mode") = dolfinx::la::BlockMode::compact)
      .def_prop_ro("dtype", [](const dolfinx::la::MatrixCSR<T>&)
                   { return dolfinx_wrappers::numpy_dtype_v<T>; })
      .def_prop_ro("bs", &dolfinx::la::MatrixCSR<T>::block_size)
      .def("squared_norm", &dolfinx::la::MatrixCSR<T>::squared_norm)
      .def("index_map", &dolfinx::la::MatrixCSR<T>::index_map)
      .def("add",
           [](dolfinx::la::MatrixCSR<T>& self, const std::vector<T>& x,
              const std::vector<std::int32_t>& rows,
              const std::vector<std::int32_t>& cols, int bs = 1)
           {
             if (bs == 1)
               self.template add<1, 1>(x, rows, cols);
             else if (bs == 2)
               self.template add<2, 2>(x, rows, cols);
             else if (bs == 3)
               self.template add<3, 3>(x, rows, cols);
             else
             {
               throw std::runtime_error(
                   "Block size not supported in this function");
             }
           })
      .def("set",
           [](dolfinx::la::MatrixCSR<T>& self, const std::vector<T>& x,
              const std::vector<std::int32_t>& rows,
              const std::vector<std::int32_t>& cols, int bs = 1)
           {
             if (bs == 1)
               self.template set<1, 1>(x, rows, cols);
             else if (bs == 2)
               self.template set<2, 2>(x, rows, cols);
             else if (bs == 3)
               self.template set<3, 3>(x, rows, cols);
             else
             {
               throw std::runtime_error(
                   "Block size not supported in this function");
             }
           })
      .def("scatter_reverse", &dolfinx::la::MatrixCSR<T>::scatter_rev)
      .def("mult", &dolfinx::la::MatrixCSR<T>::mult)
      .def("to_dense",
           [](const dolfinx::la::MatrixCSR<T>& self)
           {
             std::array<int, 2> bs = self.block_size();
             std::size_t nrows = self.num_all_rows() * bs[0];
             std::size_t ncols = self.index_map(1)->size_global() * bs[1];
             std::vector<T> dense = self.to_dense();
             assert(nrows * ncols == dense.size());
             return dolfinx_wrappers::as_nbarray(std::move(dense),
                                                 {nrows, ncols});
           })
      .def_prop_ro(
          "data",
          [](dolfinx::la::MatrixCSR<T>& self)
          {
            return nb::ndarray<T, nb::numpy>(self.values().data(),
                                             {self.values().size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "indices",
          [](dolfinx::la::MatrixCSR<T>& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.cols().data(), {self.cols().size()});
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "indptr",
          [](dolfinx::la::MatrixCSR<T>& self)
          {
            return nb::ndarray<const std::int64_t, nb::numpy>(
                self.row_ptr().data(), {self.row_ptr().size()});
          },
          nb::rv_policy::reference_internal)
      .def("scatter_rev_begin", &dolfinx::la::MatrixCSR<T>::scatter_rev_begin)
      .def("scatter_rev_end", &dolfinx::la::MatrixCSR<T>::scatter_rev_end);
}

/// Declare linear algebra functions (norm, inner_product, orthonormalize) for a
/// given scalar type
/// @param m The nanobind module
template <typename T>
void declare_la_functions(nanobind::module_& m)
{
  namespace nb = nanobind;
  using namespace nb::literals;

  m.def(
      "norm", [](const dolfinx::la::Vector<T>& x, dolfinx::la::Norm type)
      { return dolfinx::la::norm(x, type); }, "vector"_a, "type"_a);
  m.def(
      "inner_product",
      [](const dolfinx::la::Vector<T>& x, const dolfinx::la::Vector<T>& y)
      { return dolfinx::la::inner_product(x, y); }, nb::arg("x"), nb::arg("y"));
  m.def(
      "orthonormalize",
      [](std::vector<dolfinx::la::Vector<T>*> basis)
      {
        std::vector<std::reference_wrapper<dolfinx::la::Vector<T>>> _basis;
        _basis.reserve(basis.size());
        for (std::size_t i = 0; i < basis.size(); ++i)
          _basis.push_back(*basis[i]);
        dolfinx::la::orthonormalize(_basis);
      },
      nb::arg("basis"));
  m.def(
      "is_orthonormal",
      [](std::vector<const dolfinx::la::Vector<T>*> basis,
         dolfinx::scalar_value_t<T> eps)
      {
        std::vector<std::reference_wrapper<const dolfinx::la::Vector<T>>>
            _basis;
        _basis.reserve(basis.size());
        for (std::size_t i = 0; i < basis.size(); ++i)
          _basis.push_back(*basis[i]);
        return dolfinx::la::is_orthonormal(_basis, eps);
      },
      nb::arg("basis"), nb::arg("eps"));
}

#if defined(HAS_SUPERLU_DIST)
/// Declare SuperLU_DIST matrix wrapper for a given scalar type
/// @param m The nanobind module
/// @param type String representation of the scalar type (e.g., "float64",
/// "complex128")
template <typename T>
void declare_superlu_dist_matrix(nanobind::module_& m, const std::string& type)
{
  namespace nb = nanobind;
  using namespace nb::literals;

  // dolfinx::la::SuperLUDistMatrix
  std::string name = std::string("SuperLUDistMatrix_") + type;
  nb::class_<dolfinx::la::SuperLUDistMatrix<T>>(m, name.c_str())
      .def(
          "__init__",
          [](dolfinx::la::SuperLUDistMatrix<T>* Amat_superlu,
             const dolfinx::la::MatrixCSR<T>& Amat)
          { new (Amat_superlu) dolfinx::la::SuperLUDistMatrix<T>(Amat); },
          nb::arg("A"))
      .def_prop_ro("dtype", [](const dolfinx::la::SuperLUDistMatrix<T>&)
                   { return dolfinx_wrappers::numpy_dtype_v<T>; });
  ;
}

/// Declare SuperLU_DIST solver wrapper for a given scalar type
/// @param m The nanobind module
/// @param type String representation of the scalar type (e.g., "float64",
/// "complex128")
template <typename T>
void declare_superlu_dist_solver(nanobind::module_& m, const std::string& type)
{
  namespace nb = nanobind;
  using namespace nb::literals;

  // dolfinx::la::SuperLUDistSolver
  std::string name = std::string("SuperLUDistSolver_") + type;
  nb::class_<dolfinx::la::SuperLUDistSolver<T>>(m, name.c_str())
      .def(
          "__init__",
          [](dolfinx::la::SuperLUDistSolver<T>* solver,
             std::shared_ptr<const dolfinx::la::SuperLUDistMatrix<T>>
                 Amat_superlu)
          {
            new (solver)
                dolfinx::la::SuperLUDistSolver<T>(std::move(Amat_superlu));
          },
          nb::arg("A"))
      .def("set_option", &dolfinx::la::SuperLUDistSolver<T>::set_option)
      .def("set_A", &dolfinx::la::SuperLUDistSolver<T>::set_A)
      .def("solve", &dolfinx::la::SuperLUDistSolver<T>::solve);
}
#endif // HAS_SUPERLU_DIST

} // namespace dolfinx_wrappers

// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include "numpy_dtype.h"
#include <complex>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/la/utils.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <span>

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

// InsertMode types
enum class PyInsertMode
{
  add,
  insert
};

// Declare objects that have multiple scalar types
template <typename T>
void declare_objects(nb::module_& m, const std::string& type)
{
  // dolfinx::la::Vector
  std::string pyclass_vector_name = std::string("Vector_") + type;
  nb::class_<dolfinx::la::Vector<T>>(m, pyclass_vector_name.c_str())
      .def(nb::init<std::shared_ptr<const dolfinx::common::IndexMap>, int>(),
           nb::arg("map"), nb::arg("bs"))
      .def(nb::init<const dolfinx::la::Vector<T>&>(), nb::arg("vec"))
      .def_prop_ro("dtype", [](const dolfinx::la::Vector<T>&)
                   { return dolfinx_wrappers::numpy_dtype<T>(); })
      .def_prop_ro("index_map", &dolfinx::la::Vector<T>::index_map)
      .def_prop_ro("bs", &dolfinx::la::Vector<T>::bs)
      .def_prop_ro(
          "array",
          [](dolfinx::la::Vector<T>& self)
          {
            return nb::ndarray<T, nb::numpy>(self.mutable_array().data(),
                                             {self.array().size()},
                                             nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def("scatter_forward", &dolfinx::la::Vector<T>::scatter_fwd)
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
  nb::class_<dolfinx::la::MatrixCSR<T>>(m, pyclass_matrix_name.c_str())
      .def(nb::init<const dolfinx::la::SparsityPattern&,
                    dolfinx::la::BlockMode>(),
           nb::arg("p"),
           nb::arg("block_mode") = dolfinx::la::BlockMode::compact)
      .def_prop_ro("dtype", [](const dolfinx::la::MatrixCSR<T>&)
                   { return dolfinx_wrappers::numpy_dtype<T>(); })
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
      .def("set_value",
           static_cast<void (dolfinx::la::MatrixCSR<T>::*)(T)>(
               &dolfinx::la::MatrixCSR<T>::set),
           nb::arg("x"))
      .def("scatter_reverse", &dolfinx::la::MatrixCSR<T>::scatter_rev)
      .def("to_dense",
           [](const dolfinx::la::MatrixCSR<T>& self)
           {
             const std::array<int, 2> bs = self.block_size();
             std::size_t nrows = self.num_all_rows() * bs[0];
             auto map_col = self.index_map(1);
             std::size_t ncols
                 = (map_col->size_local() + map_col->num_ghosts()) * bs[1];
             return dolfinx_wrappers::as_nbarray(self.to_dense(),
                                                 {nrows, ncols});
           })
      .def_prop_ro(
          "data",
          [](dolfinx::la::MatrixCSR<T>& self)
          {
            return nb::ndarray<T, nb::numpy>(
                self.values().data(), {self.values().size()}, nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "indices",
          [](dolfinx::la::MatrixCSR<T>& self)
          {
            return nb::ndarray<const std::int32_t, nb::numpy>(
                self.cols().data(), {self.cols().size()}, nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def_prop_ro(
          "indptr",
          [](dolfinx::la::MatrixCSR<T>& self)
          {
            std::span<const std::int64_t> array = self.row_ptr();
            return nb::ndarray<const std::int64_t, nb::numpy>(
                array.data(), {array.size()}, nb::handle());
          },
          nb::rv_policy::reference_internal)
      .def("scatter_rev_begin", &dolfinx::la::MatrixCSR<T>::scatter_rev_begin)
      .def("scatter_rev_end", &dolfinx::la::MatrixCSR<T>::scatter_rev_end);
}

// Declare objects that have multiple scalar types
template <typename T>
void declare_functions(nb::module_& m)
{
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
        for (std::size_t i = 0; i < basis.size(); ++i)
          _basis.push_back(*basis[i]);
        dolfinx::la::orthonormalize(_basis);
      },
      nb::arg("basis"));
  m.def(
      "is_orthonormal",
      [](std::vector<const dolfinx::la::Vector<T>*> basis,
         dolfinx::scalar_value_type_t<T> eps)
      {
        std::vector<std::reference_wrapper<const dolfinx::la::Vector<T>>>
            _basis;
        for (std::size_t i = 0; i < basis.size(); ++i)
          _basis.push_back(*basis[i]);
        return dolfinx::la::is_orthonormal(_basis, eps);
      },
      nb::arg("basis"), nb::arg("eps"));
}

} // namespace

namespace dolfinx_wrappers
{
void la(nb::module_& m)
{
  nb::enum_<PyInsertMode>(m, "InsertMode")
      .value("add", PyInsertMode::add)
      .value("insert", PyInsertMode::insert);

  nb::enum_<dolfinx::la::BlockMode>(m, "BlockMode")
      .value("compact", dolfinx::la::BlockMode::compact)
      .value("expanded", dolfinx::la::BlockMode::expanded);

  nb::enum_<dolfinx::la::Norm>(m, "Norm")
      .value("l1", dolfinx::la::Norm::l1, "l1 norm")
      .value("l2", dolfinx::la::Norm::l2, "l2 norm")
      .value("linf", dolfinx::la::Norm::linf, "linf norm")
      .value("frobenius", dolfinx::la::Norm::frobenius, "Frobenius norm");

  // dolfinx::la::SparsityPattern
  nb::class_<dolfinx::la::SparsityPattern>(m, "SparsityPattern")
      .def(
          "__init__",
          [](dolfinx::la::SparsityPattern* sp, MPICommWrapper comm,
             const std::array<std::shared_ptr<const dolfinx::common::IndexMap>,
                              2>& maps,
             const std::array<int, 2>& bs)
          { new (sp) dolfinx::la::SparsityPattern(comm.get(), maps, bs); },
          nb::arg("comm"), nb::arg("maps"), nb::arg("bs"))
      .def(
          "__init__",
          [](dolfinx::la::SparsityPattern* sp, MPICommWrapper comm,
             const std::vector<std::vector<const dolfinx::la::SparsityPattern*>>
                 patterns,
             const std::array<
                 std::vector<std::pair<
                     std::reference_wrapper<const dolfinx::common::IndexMap>,
                     int>>,
                 2>& maps,
             const std::array<std::vector<int>, 2>& bs) {
            new (sp)
                dolfinx::la::SparsityPattern(comm.get(), patterns, maps, bs);
          },
          nb::arg("comm"), nb::arg("patterns"), nb::arg("maps"), nb::arg("bs"))
      .def("index_map", &dolfinx::la::SparsityPattern::index_map,
           nb::arg("dim"))
      .def("column_index_map", &dolfinx::la::SparsityPattern::column_index_map)
      .def("finalize", &dolfinx::la::SparsityPattern::finalize)
      .def_prop_ro("num_nonzeros", &dolfinx::la::SparsityPattern::num_nonzeros)
      .def(
          "insert",
          [](dolfinx::la::SparsityPattern& self,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> rows,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> cols)
          {
            self.insert(std::span(rows.data(), rows.size()),
                        std::span(cols.data(), cols.size()));
          },
          nb::arg("rows"), nb::arg("cols"))
      .def("insert",
           nb::overload_cast<int32_t, int32_t>(
               &dolfinx::la::SparsityPattern::insert),
           nb::arg("row"), nb::arg("col"))
      .def(
          "insert_diagonal",
          [](dolfinx::la::SparsityPattern& self,
             nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> rows)
          { self.insert_diagonal(std::span(rows.data(), rows.size())); },
          nb::arg("rows"))
      .def_prop_ro(
          "graph",
          [](dolfinx::la::SparsityPattern& self)
          {
            auto [edges, ptr] = self.graph();
            return std::pair(nb::ndarray<const std::int32_t, nb::numpy>(
                                 edges.data(), {edges.size()}, nb::handle()),
                             nb::ndarray<const std::int64_t, nb::numpy>(
                                 ptr.data(), {ptr.size()}, nb::handle()));
          },
          nb::rv_policy::reference_internal);

  // Declare objects that are templated over type
  declare_objects<std::int8_t>(m, "int8");
  declare_objects<std::int32_t>(m, "int32");
  declare_objects<std::int64_t>(m, "int64");
  declare_objects<float>(m, "float32");
  declare_objects<double>(m, "float64");
  declare_objects<std::complex<float>>(m, "complex64");
  declare_objects<std::complex<double>>(m, "complex128");

  declare_functions<float>(m);
  declare_functions<double>(m);
  declare_functions<std::complex<float>>(m);
  declare_functions<std::complex<double>>(m);
}
} // namespace dolfinx_wrappers

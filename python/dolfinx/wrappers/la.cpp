// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_mpi.h"
#include "caster_petsc.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/la/utils.h>
#include <memory>
#include <petsc4py/petsc4py.h>
// #include <nanobind/complex.h>
#include <nanobind/ndarray.h>
// #include <nanobind/operators.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <span>

namespace nb = nanobind;

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
      .def(
          "__init__",
          [](dolfinx::la::Vector<T>* v,
             std::shared_ptr<const dolfinx::common::IndexMap> map, int bs)
          { new (v) dolfinx::la::Vector<T>(map, bs); },
          nb::arg("map"), nb::arg("bs"))
      .def(
          "__init__",
          [](dolfinx::la::Vector<T>* v, const dolfinx::la::Vector<T>& vec)
          { new (v) dolfinx::la::Vector<T>(vec); },
          nb::arg("vec"))
      .def_prop_ro("dtype", [](const dolfinx::la::Vector<T>& self)
                   { return nb::dtype<T>(); })
      .def("set", &dolfinx::la::Vector<T>::set, nb::arg("v"))
      .def(
          "norm",
          [](dolfinx::la::Vector<T>& self, dolfinx::la::Norm type)
          { return dolfinx::la::norm(self, type); },
          nb::arg("type") = dolfinx::la::Norm::l2)
      .def_prop_ro("index_map", &dolfinx::la::Vector<T>::index_map)
      .def_prop_ro("bs", &dolfinx::la::Vector<T>::bs)
      .def_prop_ro("array",
                   [](dolfinx::la::Vector<T>& self)
                   {
                     std::span<T> array = self.mutable_array();
                     const std::size_t size = array.size();
                     return nb::ndarray<T>(array.data(), 1, &size,
                                           nb::cast(self));
                   })
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
      .def(
          "__init__",
          [](dolfinx::la::MatrixCSR<T>* mat,
             const dolfinx::la::SparsityPattern& p, dolfinx::la::BlockMode bm)
          { new (mat) dolfinx::la::MatrixCSR<T>(p, bm); },
          nb::arg("p"), nb::arg("block_mode") = dolfinx::la::BlockMode::compact)
      .def_prop_ro("dtype", [](const dolfinx::la::MatrixCSR<T>& self)
                   { return nb::dtype<T>(); })
      .def_prop_ro("block_size", &dolfinx::la::MatrixCSR<T>::block_size)
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
               throw std::runtime_error(
                   "Block size not supported in this function");
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
               throw std::runtime_error(
                   "Block size not supported in this function");
           })
      .def("set",
           static_cast<void (dolfinx::la::MatrixCSR<T>::*)(T)>(
               &dolfinx::la::MatrixCSR<T>::set),
           nb::arg("x"))
      .def("finalize", &dolfinx::la::MatrixCSR<T>::finalize)
      .def("to_dense",
           [](const dolfinx::la::MatrixCSR<T>& self)
           {
             const std::array<int, 2> bs = self.block_size();
             std::size_t nrows = self.num_all_rows() * bs[0];
             auto map_col = self.index_map(1);
             std::size_t ncols
                 = (map_col->size_local() + map_col->num_ghosts()) * bs[1];
             return dolfinx_wrappers::as_nbarray(self.to_dense(),
                                                 std::array{nrows, ncols});
           })
      .def_prop_ro("data",
                   [](dolfinx::la::MatrixCSR<T>& self)
                   {
                     std::span<T> array = self.values();
                     const std::size_t size = array.size();
                     return nb::ndarray<const T>(array.data(), 1, &size,
                                           nb::cast(self));
                   })
      .def_prop_ro("indices",
                   [](dolfinx::la::MatrixCSR<T>& self)
                   {
                     auto& array = self.cols();
                     const std::size_t size = array.size();
                     return nb::ndarray<const std::int64_t>(array.data(), 1, &size, nb::cast(self));
                   })
      .def_prop_ro("indptr",
                   [](dolfinx::la::MatrixCSR<T>& self)
                   {
                     auto& array = self.row_ptr();
                     const std::size_t size = array.size();
                     return nb::ndarray<const std::int32_t>(array.data(), 1, &size, nb::cast(self));
                   })
      .def("finalize_begin", &dolfinx::la::MatrixCSR<T>::finalize_begin)
      .def("finalize_end", &dolfinx::la::MatrixCSR<T>::finalize_end);
}

void petsc_module(nb::module_& m)
{
  m.def("create_vector",
        nb::overload_cast<const dolfinx::common::IndexMap&, int>(
            &dolfinx::la::petsc::create_vector),
        nb::rv_policy::take_ownership, nb::arg("index_map"), nb::arg("bs"),
        "Create a ghosted PETSc Vec for index map.");
  m.def(
      "create_vector_wrap",
      [](dolfinx::la::Vector<PetscScalar>& x)
      { return dolfinx::la::petsc::create_vector_wrap(x); },
      nb::rv_policy::take_ownership, nb::arg("x"),
      "Create a ghosted PETSc Vec that wraps a DOLFINx Vector");
  m.def(
      "create_matrix",
      [](dolfinx_wrappers::MPICommWrapper comm,
         const dolfinx::la::SparsityPattern& p, const std::string& type)
      { return dolfinx::la::petsc::create_matrix(comm.get(), p, type); },
      nb::rv_policy::take_ownership, nb::arg("comm"), nb::arg("p"),
      nb::arg("type") = std::string(),
      "Create a PETSc Mat from sparsity pattern.");

  // TODO: check reference counting for index sets
  m.def("create_index_sets", &dolfinx::la::petsc::create_index_sets,
        nb::arg("maps"), nb::rv_policy::take_ownership);

  m.def(
      "scatter_local_vectors",
      [](Vec x, const std::vector<nb::ndarray<PetscScalar, nb::numpy>>& x_b,
         const std::vector<std::pair<
             std::reference_wrapper<const dolfinx::common::IndexMap>, int>>&
             maps)
      {
        std::vector<std::span<const PetscScalar>> _x_b;
        for (auto& array : x_b)
          _x_b.emplace_back(array.data(), array.size());
        dolfinx::la::petsc::scatter_local_vectors(x, _x_b, maps);
      },
      nb::arg("x"), nb::arg("x_b"), nb::arg("maps"),
      "Scatter the (ordered) list of sub vectors into a block "
      "vector.");
  m.def(
      "get_local_vectors",
      [](const Vec x,
         const std::vector<std::pair<
             std::reference_wrapper<const dolfinx::common::IndexMap>, int>>&
             maps)
      {
        std::vector<std::vector<PetscScalar>> vecs
            = dolfinx::la::petsc::get_local_vectors(x, maps);
        std::vector<nb::ndarray<PetscScalar>> ret;
        for (std::vector<PetscScalar>& v : vecs)
          ret.push_back(dolfinx_wrappers::as_nbarray(std::move(v)));
        return ret;
      },
      nb::arg("x"), nb::arg("maps"),
      "Gather an (ordered) list of sub vectors from a block vector.");
}

} // namespace

namespace dolfinx_wrappers
{
void la(nb::module_& m)
{
  nb::module_ petsc_mod
      = m.def_submodule("petsc", "PETSc-specific linear algebra");
  // petsc_module(petsc_mod);

  nb::enum_<PyInsertMode>(m, "InsertMode")
      .value("add", PyInsertMode::add)
      .value("insert", PyInsertMode::insert);

  nb::enum_<dolfinx::la::BlockMode>(m, "BlockMode")
      .value("compact", dolfinx::la::BlockMode::compact)
      .value("expanded", dolfinx::la::BlockMode::expanded);

  nb::enum_<dolfinx::la::Norm>(m, "Norm")
      .value("l1", dolfinx::la::Norm::l1)
      .value("l2", dolfinx::la::Norm::l2)
      .value("linf", dolfinx::la::Norm::linf)
      .value("frobenius", dolfinx::la::Norm::frobenius);

  // dolfinx::la::SparsityPattern
  nb::class_<dolfinx::la::SparsityPattern>(m, "SparsityPattern")
      .def(
          "__init__",
          [](dolfinx::la::SparsityPattern* sp, const MPICommWrapper comm,
             const std::array<std::shared_ptr<const dolfinx::common::IndexMap>,
                              2>& maps,
             const std::array<int, 2>& bs)
          { new (sp) dolfinx::la::SparsityPattern(comm.get(), maps, bs); },
          nb::arg("comm"), nb::arg("maps"), nb::arg("bs"))
      .def(
          "__init__",
          [](dolfinx::la::SparsityPattern* sp, const MPICommWrapper comm,
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
             const nb::ndarray<std::int32_t, nb::numpy>& rows,
             const nb::ndarray<std::int32_t, nb::numpy>& cols)
          {
            self.insert(std::span(rows.data(), rows.size()),
                        std::span(cols.data(), cols.size()));
          },
          nb::arg("rows"), nb::arg("cols"))
      .def(
          "insert_diagonal",
          [](dolfinx::la::SparsityPattern& self,
             const nb::ndarray<std::int32_t, nb::numpy>& rows)
          { self.insert_diagonal(std::span(rows.data(), rows.size())); },
          nb::arg("rows"))
      .def_prop_ro(
          "graph",
          [](dolfinx::la::SparsityPattern& self)
          {
            auto [edges, ptr] = self.graph();
            const std::size_t esize = edges.size();
            const std::size_t psize = ptr.size();

            return std::pair(
                nb::ndarray<const std::int32_t>(edges.data(), 1, &esize, nb::cast(self)),
                nb::ndarray<const std::int64_t>(ptr.data(), 1, &psize, nb::cast(self)));
          });

  // Declare objects that are templated over type
  declare_objects<float>(m, "float32");
  declare_objects<double>(m, "float64");


  //  declare_objects<std::complex<float>>(m, "complex64");
  //  declare_objects<std::complex<double>>(m, "complex128");
}
} // namespace dolfinx_wrappers

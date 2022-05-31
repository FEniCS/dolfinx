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
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtl/xspan.hpp>

namespace py = pybind11;

namespace
{
// ScatterMode types
enum class PyScatterMode
{
  add,
  insert
};

// Declare objects that have multiple scalar types
template <typename T>
void declare_objects(py::module& m, const std::string& type)
{
  // dolfinx::la::Vector
  std::string pyclass_vector_name = std::string("Vector_") + type;
  py::class_<dolfinx::la::Vector<T>, std::shared_ptr<dolfinx::la::Vector<T>>>(
      m, pyclass_vector_name.c_str())
      .def(py::init(
          [](const std::shared_ptr<const dolfinx::common::IndexMap>& map,
             int bs) { return dolfinx::la::Vector<T>(map, bs); }))
      .def(py::init([](const dolfinx::la::Vector<T>& vec)
                    { return dolfinx::la::Vector<T>(vec); }))
      .def_property_readonly("dtype", [](const dolfinx::la::Vector<T>& self)
                             { return py::dtype::of<T>(); })
      .def("set", &dolfinx::la::Vector<T>::set)
      .def(
          "norm",
          [](dolfinx::la::Vector<T>& self, dolfinx::la::Norm type)
          { return dolfinx::la::norm(self, type); },
          py::arg("type") = dolfinx::la::Norm::l2)
      .def_property_readonly("map", &dolfinx::la::Vector<T>::map)
      .def_property_readonly("bs", &dolfinx::la::Vector<T>::bs)
      .def_property_readonly("array",
                             [](dolfinx::la::Vector<T>& self)
                             {
                               xtl::span<T> array = self.mutable_array();
                               return py::array_t<T>(array.size(), array.data(),
                                                     py::cast(self));
                             })
      .def("scatter_forward", &dolfinx::la::Vector<T>::scatter_fwd)
      .def("scatter_reverse",
           [](dolfinx::la::Vector<T>& self, PyScatterMode mode)
           {
             switch (mode)
             {
             case PyScatterMode::add: // Add
               self.scatter_rev(std::plus<T>());
               break;
             case PyScatterMode::insert: // Insert
               self.scatter_rev([](T /*a*/, T b) { return b; });
               break;
             default:
               throw std::runtime_error("ScatterMode not recognized.");
               break;
             }
           });

  // dolfinx::la::MatrixCSR
  std::string pyclass_matrix_name = std::string("MatrixCSR_") + type;
  py::class_<dolfinx::la::MatrixCSR<T>,
             std::shared_ptr<dolfinx::la::MatrixCSR<T>>>(
      m, pyclass_matrix_name.c_str())
      .def(py::init([](const dolfinx::la::SparsityPattern& p)
                    { return dolfinx::la::MatrixCSR<T>(p); }))
      .def_property_readonly("dtype", [](const dolfinx::la::MatrixCSR<T>& self)
                             { return py::dtype::of<T>(); })
      .def("norm_squared", &dolfinx::la::MatrixCSR<T>::norm_squared)
      .def("mat_add_values", &dolfinx::la::MatrixCSR<T>::mat_add_values)
      .def("set", static_cast<void (dolfinx::la::MatrixCSR<T>::*)(T)>(
                      &dolfinx::la::MatrixCSR<T>::set))
      .def("finalize", &dolfinx::la::MatrixCSR<T>::finalize)
      .def("to_dense",
           [](const dolfinx::la::MatrixCSR<T>& self)
           {
             std::size_t nrows = self.num_all_rows();
             auto map_col = self.index_maps()[1];
             std::size_t ncols = map_col->size_local() + map_col->num_ghosts();
             return dolfinx_wrappers::as_pyarray(self.to_dense(),
                                                 std::array{nrows, ncols});
           })
      .def_property_readonly("data",
                             [](dolfinx::la::MatrixCSR<T>& self)
                             {
                               xtl::span<T> array = self.values();
                               return py::array_t<T>(array.size(), array.data(),
                                                     py::cast(self));
                             })
      .def_property_readonly("indices",
                             [](dolfinx::la::MatrixCSR<T>& self)
                             {
                               xtl::span<const std::int32_t> array
                                   = self.cols();
                               return py::array_t<const std::int32_t>(
                                   array.size(), array.data(), py::cast(self));
                             })
      .def_property_readonly("indptr",
                             [](dolfinx::la::MatrixCSR<T>& self)
                             {
                               xtl::span<const std::int32_t> array
                                   = self.row_ptr();
                               return py::array_t<const std::int32_t>(
                                   array.size(), array.data(), py::cast(self));
                             })
      .def("finalize_begin", &dolfinx::la::MatrixCSR<T>::finalize_begin)
      .def("finalize_end", &dolfinx::la::MatrixCSR<T>::finalize_end);
}

void petsc_module(py::module& m)
{
  m.def("create_vector",
        py::overload_cast<const dolfinx::common::IndexMap&, int>(
            &dolfinx::la::petsc::create_vector),
        py::return_value_policy::take_ownership,
        "Create a ghosted PETSc Vec for index map.");
  m.def(
      "create_vector_wrap",
      [](dolfinx::la::Vector<PetscScalar, std::allocator<PetscScalar>>& x)
      { return dolfinx::la::petsc::create_vector_wrap(x); },
      py::return_value_policy::take_ownership,
      "Create a ghosted PETSc Vec that wraps a DOLFINx Vector");
  m.def(
      "create_matrix",
      [](dolfinx_wrappers::MPICommWrapper comm,
         const dolfinx::la::SparsityPattern& p, const std::string& type)
      { return dolfinx::la::petsc::create_matrix(comm.get(), p, type); },
      py::return_value_policy::take_ownership, py::arg("comm"), py::arg("p"),
      py::arg("type") = std::string(),
      "Create a PETSc Mat from sparsity pattern.");

  // TODO: check reference counting for index sets
  m.def("create_index_sets", &dolfinx::la::petsc::create_index_sets,
        py::return_value_policy::take_ownership);

  m.def(
      "scatter_local_vectors",
      [](Vec x,
         const std::vector<py::array_t<PetscScalar, py::array::c_style>>& x_b,
         const std::vector<std::pair<
             std::reference_wrapper<const dolfinx::common::IndexMap>, int>>&
             maps)
      {
        std::vector<xtl::span<const PetscScalar>> _x_b;
        for (auto& array : x_b)
          _x_b.emplace_back(array.data(), array.size());
        dolfinx::la::petsc::scatter_local_vectors(x, _x_b, maps);
      },
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
        std::vector<py::array> ret;
        for (std::vector<PetscScalar>& v : vecs)
          ret.push_back(dolfinx_wrappers::as_pyarray(std::move(v)));
        return ret;
      },
      "Gather an (ordered) list of sub vectors from a block vector.");
}

} // namespace

namespace dolfinx_wrappers
{
void la(py::module& m)
{
  py::module petsc_mod
      = m.def_submodule("petsc", "PETSc-specific linear algebra");
  petsc_module(petsc_mod);

  py::enum_<PyScatterMode>(m, "ScatterMode")
      .value("add", PyScatterMode::add)
      .value("insert", PyScatterMode::insert);

  py::enum_<dolfinx::la::Norm>(m, "Norm")
      .value("l1", dolfinx::la::Norm::l1)
      .value("l2", dolfinx::la::Norm::l2)
      .value("linf", dolfinx::la::Norm::linf)
      .value("frobenius", dolfinx::la::Norm::frobenius);

  // dolfinx::la::SparsityPattern
  py::class_<dolfinx::la::SparsityPattern,
             std::shared_ptr<dolfinx::la::SparsityPattern>>(m,
                                                            "SparsityPattern")
      .def(py::init(
          [](const MPICommWrapper comm,
             const std::array<std::shared_ptr<const dolfinx::common::IndexMap>,
                              2>& maps,
             const std::array<int, 2>& bs)
          { return dolfinx::la::SparsityPattern(comm.get(), maps, bs); }))
      .def(py::init(
          [](const MPICommWrapper comm,
             const std::vector<std::vector<const dolfinx::la::SparsityPattern*>>
                 patterns,
             const std::array<
                 std::vector<std::pair<
                     std::reference_wrapper<const dolfinx::common::IndexMap>,
                     int>>,
                 2>& maps,
             const std::array<std::vector<int>, 2>& bs) {
            return dolfinx::la::SparsityPattern(comm.get(), patterns, maps, bs);
          }))
      .def("index_map", &dolfinx::la::SparsityPattern::index_map)
      .def("column_index_map", &dolfinx::la::SparsityPattern::column_index_map)
      .def("assemble", &dolfinx::la::SparsityPattern::assemble)
      .def_property_readonly("num_nonzeros",
                             &dolfinx::la::SparsityPattern::num_nonzeros)
      .def("insert",
           [](dolfinx::la::SparsityPattern& self,
              const py::array_t<std::int32_t, py::array::c_style>& rows,
              const py::array_t<std::int32_t, py::array::c_style>& cols)
           {
             self.insert(xtl::span(rows.data(), rows.size()),
                         xtl::span(cols.data(), cols.size()));
           })
      .def("insert_diagonal",
           [](dolfinx::la::SparsityPattern& self,
              const py::array_t<std::int32_t, py::array::c_style>& rows)
           { self.insert_diagonal(rows); })
      .def_property_readonly("graph", &dolfinx::la::SparsityPattern::graph,
                             py::return_value_policy::reference_internal);

  // Declare objects that are templated over type
  declare_objects<double>(m, "float64");
  declare_objects<float>(m, "float32");
  declare_objects<std::complex<double>>(m, "complex128");
  declare_objects<std::complex<float>>(m, "complex64");
}
} // namespace dolfinx_wrappers

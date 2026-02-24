// Copyright (C) 2017-2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dolfinx_wrappers/la.h"
#include "dolfinx_wrappers/caster_mpi.h"
#include <complex>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/superlu_dist.h>
#include <dolfinx/la/utils.h>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <span>

#if defined(HAS_SUPERLU_DIST)
#include <superlu_defs.h>
struct dolfinx::la::SuperLUDistStructs::vec_int_t
{
  /// @brief vector
  std::vector<int_t> vec;
};
struct dolfinx::la::SuperLUDistStructs::superlu_dist_options_t
    : public ::superlu_dist_options_t
{
};
#endif // HAS_SUPERLU_DIST

namespace nb = nanobind;
using namespace nb::literals;

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
             std::array<std::shared_ptr<const dolfinx::common::IndexMap>, 2>
                 maps,
             std::array<int, 2> bs)
          { new (sp) dolfinx::la::SparsityPattern(comm.get(), maps, bs); },
          nb::arg("comm"), nb::arg("maps"), nb::arg("bs"))
      .def(
          "__init__",
          [](dolfinx::la::SparsityPattern* sp, MPICommWrapper comm,
             const std::vector<
                 std::vector<const dolfinx::la::SparsityPattern*>>& patterns,
             const std::array<
                 std::vector<std::pair<
                     std::reference_wrapper<const dolfinx::common::IndexMap>,
                     int>>,
                 2>& maps,
             std::array<std::vector<int>, 2> bs)
          {
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
                                 edges.data(), {edges.size()}),
                             nb::ndarray<const std::int64_t, nb::numpy>(
                                 ptr.data(), {ptr.size()}));
          },
          nb::rv_policy::reference_internal);

#if defined(HAS_SUPERLU_DIST)
  declare_superlu_dist_matrix<double>(m, "float64");
  declare_superlu_dist_matrix<float>(m, "float32");
  declare_superlu_dist_matrix<std::complex<double>>(m, "complex128");

  declare_superlu_dist_solver<double>(m, "float64");
  declare_superlu_dist_solver<float>(m, "float32");
  declare_superlu_dist_solver<std::complex<double>>(m, "complex128");
#endif // HAS_SUPERLU_DIST

  // Declare objects that are templated over type
  declare_la_objects<std::int8_t>(m, "int8");
  declare_la_objects<std::int32_t>(m, "int32");
  declare_la_objects<std::int64_t>(m, "int64");
  declare_la_objects<float>(m, "float32");
  declare_la_objects<double>(m, "float64");
  declare_la_objects<std::complex<float>>(m, "complex64");
  declare_la_objects<std::complex<double>>(m, "complex128");

  declare_la_functions<float>(m);
  declare_la_functions<double>(m);
  declare_la_functions<std::complex<float>>(m);
  declare_la_functions<std::complex<double>>(m);
}
} // namespace dolfinx_wrappers

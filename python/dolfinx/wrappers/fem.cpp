// Copyright (C) 2017-2026 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dolfinx_wrappers/fem.h"
#include "dolfinx_wrappers/array.h"
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/dofmapbuilder.h>
#include <dolfinx/fem/sparsitybuild.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/graph/ordering.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <functional>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <span>
#include <string>
#include <tuple>
#include <utility>

namespace nb = nanobind;
namespace md = MDSPAN_IMPL_STANDARD_NAMESPACE;

namespace dolfinx_wrappers
{
void fem(nb::module_& m)
{
  m.def(
      "build_dofmap",
      [](MPICommWrapper comm, const dolfinx::mesh::Topology& topology,
         const dolfinx::fem::ElementDofLayout& layout)
      {
        assert(topology.entity_types(topology.dim()).size() == 1);
        auto [map, bs, dofmap] = dolfinx::fem::build_dofmap_data(
            comm.get(), topology, {layout},
            [](const dolfinx::graph::AdjacencyList<std::int32_t>& g)
            { return dolfinx::graph::reorder_gps(g); });
        return std::tuple(std::move(map), bs, std::move(dofmap));
      },
      nb::arg("comm"), nb::arg("topology"), nb::arg("layout"),
      "Build a dofmap on a mesh.");
  m.def(
      "transpose_dofmap",
      [](nb::ndarray<const std::int32_t, nb::ndim<2>, nb::c_contig> dofmap,
         int num_cells)
      {
        md::mdspan<const std::int32_t, md::dextents<std::size_t, 2>> _dofmap(
            dofmap.data(), dofmap.shape(0), dofmap.shape(1));
        return dolfinx::fem::transpose_dofmap(_dofmap, num_cells);
      },
      "Build the index to (cell, local index) map from a dofmap ((cell, local "
      "index) -> index).");
  m.def(
      "compute_integration_domains",
      [](dolfinx::fem::IntegralType type,
         const dolfinx::mesh::Topology& topology,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities)
      {
        return dolfinx_wrappers::as_nbarray(
            dolfinx::fem::compute_integration_domains(
                type, topology, std::span(entities.data(), entities.size())));
      },
      nb::arg("integral_type"), nb::arg("topology"), nb::arg("entities"));

  // dolfinx::fem::ElementDofLayout
  nb::class_<dolfinx::fem::ElementDofLayout>(
      m, "ElementDofLayout", "Object describing the layout of dofs on a cell")
      .def(nb::init<int, const std::vector<std::vector<std::vector<int>>>&,
                    const std::vector<std::vector<std::vector<int>>>&,
                    const std::vector<int>&,
                    const std::vector<dolfinx::fem::ElementDofLayout>&>(),
           nb::arg("block_size"), nb::arg("entity_dofs"),
           nb::arg("entity_closure_dofs"), nb::arg("parent_map"),
           nb::arg("sub_layouts"))
      .def_prop_ro("num_dofs", &dolfinx::fem::ElementDofLayout::num_dofs)
      .def("entity_dofs", &dolfinx::fem::ElementDofLayout::entity_dofs,
           nb::arg("dim"), nb::arg("entity_index"))
      .def("entity_closure_dofs",
           &dolfinx::fem::ElementDofLayout::entity_closure_dofs, nb::arg("dim"),
           nb::arg("entity_index"))
      .def_prop_ro("block_size", &dolfinx::fem::ElementDofLayout::block_size);

  // dolfinx::fem::DofMap
  nb::class_<dolfinx::fem::DofMap>(m, "DofMap", "DofMap object")
      .def(
          "__init__",
          [](dolfinx::fem::DofMap* self,
             const dolfinx::fem::ElementDofLayout& element,
             std::shared_ptr<const dolfinx::common::IndexMap> index_map,
             int index_map_bs,
             const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap, int bs)
          {
            new (self) dolfinx::fem::DofMap(element, index_map, index_map_bs,
                                            dofmap.array(), bs);
          },
          nb::arg("element_dof_layout"), nb::arg("index_map"),
          nb::arg("index_map_bs"), nb::arg("dofmap"), nb::arg("bs"))
      .def_ro("index_map", &dolfinx::fem::DofMap::index_map)
      .def_prop_ro("index_map_bs", &dolfinx::fem::DofMap::index_map_bs)
      .def_prop_ro("dof_layout", &dolfinx::fem::DofMap::element_dof_layout)
      .def(
          "cell_dofs",
          [](const dolfinx::fem::DofMap& self, int cell)
          {
            std::span<const std::int32_t> dofs = self.cell_dofs(cell);
            return nb::ndarray<const std::int32_t, nb::numpy>(dofs.data(),
                                                              {dofs.size()});
          },
          nb::rv_policy::reference_internal, nb::arg("cell"))
      .def_prop_ro("bs", &dolfinx::fem::DofMap::bs)
      .def(
          "map",
          [](const dolfinx::fem::DofMap& self)
          {
            auto dofs = self.map();
            return nb::ndarray<const std::int32_t, nb::numpy>(
                dofs.data_handle(), {dofs.extent(0), dofs.extent(1)});
          },
          nb::rv_policy::reference_internal);

  nb::enum_<dolfinx::fem::IntegralType>(m, "_IntegralType")
      .value("cell", dolfinx::fem::IntegralType::cell, "cell integral")
      .value("exterior_facet", dolfinx::fem::IntegralType::exterior_facet,
             "exterior facet integral")
      .value("interior_facet", dolfinx::fem::IntegralType::interior_facet,
             "exterior facet integral")
      .value("vertex", dolfinx::fem::IntegralType::vertex, "vertex integral")
      .value("ridge", dolfinx::fem::IntegralType::ridge, "ridge integral");

  declare_objects<float>(m, "float32");
  declare_objects<double>(m, "float64");
  declare_objects<std::complex<float>>(m, "complex64");
  declare_objects<std::complex<double>>(m, "complex128");

  declare_form<float>(m, "float32");
  declare_form<double>(m, "float64");
  declare_form<std::complex<float>>(m, "complex64");
  declare_form<std::complex<double>>(m, "complex128");

  declare_cmap<float>(m, "float32");
  declare_cmap<double>(m, "float64");

  declare_function_space<float>(m, "float32");
  declare_function_space<double>(m, "float64");

  declare_real_functions<float>(m);
  declare_real_functions<double>(m);
}
} // namespace dolfinx_wrappers

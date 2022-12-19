// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Geometry.h"
#include "Topology.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/fem/dofmapbuilder.h>
#include <dolfinx/graph/partition.h>

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
int Geometry::dim() const { return _dim; }
//-----------------------------------------------------------------------------
const graph::AdjacencyList<std::int32_t>& Geometry::dofmap() const
{
  return _dofmap;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const common::IndexMap> Geometry::index_map() const
{
  return _index_map;
}
//-----------------------------------------------------------------------------
std::span<double> Geometry::x() { return _x; }
//-----------------------------------------------------------------------------
std::span<const double> Geometry::x() const { return _x; }
//-----------------------------------------------------------------------------
const fem::CoordinateElement& Geometry::cmap() const { return _cmap; }
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>& Geometry::input_global_indices() const
{
  return _input_global_indices;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
mesh::Geometry mesh::create_geometry(
    MPI_Comm comm, const Topology& topology,
    const fem::CoordinateElement& element,
    const graph::AdjacencyList<std::int64_t>& cell_nodes,
    std::span<const double> x, int dim,
    const std::function<std::vector<int>(
        const graph::AdjacencyList<std::int32_t>&)>& reorder_fn)
{
  // TODO: make sure required entities are initialised, or extend
  // fem::build_dofmap_data

  //  Build 'geometry' dofmap on the topology
  auto [_dof_index_map, bs, dofmap] = fem::build_dofmap_data(
      comm, topology, element.create_dof_layout(), reorder_fn);
  auto dof_index_map
      = std::make_shared<common::IndexMap>(std::move(_dof_index_map));

  // If the mesh has higher order geometry, permute the dofmap
  if (element.needs_dof_permutations())
  {
    const int D = topology.dim();
    const int num_cells = topology.connectivity(D, 0)->num_nodes();
    const std::vector<std::uint32_t>& cell_info
        = topology.get_cell_permutation_info();

    for (std::int32_t cell = 0; cell < num_cells; ++cell)
      element.unpermute_dofs(dofmap.links(cell), cell_info[cell]);
  }

  auto remap_data
      = [](auto comm, auto& cell_nodes, auto& x, int dim, auto& dofmap)
  {
    // Build list of unique (global) node indices from adjacency list
    // (geometry nodes)
    std::vector<std::int64_t> indices = cell_nodes.array();
    dolfinx::radix_sort(std::span(indices));
    indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

    //  Distribute  node coordinates by global index from other ranks.
    //  Order of coords matches order of the indices in 'indices'.
    std::vector<double> coords
        = MPI::distribute_data<double>(comm, indices, x, dim);

    // Compute local-to-global map from local indices in dofmap to the
    // corresponding global indices in cell_nodes
    std::vector l2g
        = graph::build::compute_local_to_global_links(cell_nodes, dofmap);

    // Compute local (dof) to local (position in coords) map from (i)
    // local-to-global for dofs and (ii) local-to-global for entries in
    // coords
    std::vector l2l = graph::build::compute_local_to_local(l2g, indices);

    // Allocate space for input global indices and copy data
    std::vector<std::int64_t> igi(indices.size());
    std::transform(l2l.cbegin(), l2l.cend(), igi.begin(),
                   [&indices](auto index) { return indices[index]; });

    return std::tuple(std::move(coords), std::move(l2l), std::move(igi));
  };

  auto [coords, l2l, igi] = remap_data(comm, cell_nodes, x, dim, dofmap);

  // Build coordinate dof array, copying coordinates to correct
  // position
  assert(coords.size() % dim == 0);
  const std::size_t shape0 = coords.size() / dim;
  const std::size_t shape1 = dim;
  std::vector<double> xg(3 * shape0, 0);
  for (std::size_t i = 0; i < shape0; ++i)
  {
    std::copy_n(std::next(coords.cbegin(), shape1 * l2l[i]), shape1,
                std::next(xg.begin(), 3 * i));
  }

  return Geometry(dof_index_map, std::move(dofmap), element, std::move(xg), dim,
                  std::move(igi));
}
//-----------------------------------------------------------------------------

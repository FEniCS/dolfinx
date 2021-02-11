// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Geometry.h"
#include "Topology.h"
#include <boost/functional/hash.hpp>
#include <dolfinx/common/IndexMap.h>
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
common::array2d<double>& Geometry::x() { return _x; }
//-----------------------------------------------------------------------------
const common::array2d<double>& Geometry::x() const { return _x; }
//-----------------------------------------------------------------------------
const fem::CoordinateElement& Geometry::cmap() const { return _cmap; }
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>& Geometry::input_global_indices() const
{
  return _input_global_indices;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
mesh::Geometry
mesh::create_geometry(MPI_Comm comm, const Topology& topology,
                      const fem::CoordinateElement& coordinate_element,
                      const graph::AdjacencyList<std::int64_t>& cell_nodes,
                      const common::array2d<double>& x)
{
  // TODO: make sure required entities are initialised, or extend
  // fem::build_dofmap_data

  //  Build 'geometry' dofmap on the topology
  auto [dof_index_map, bs, dofmap]
      = fem::build_dofmap_data(comm, topology, coordinate_element.dof_layout());

  // If the mesh has higher order geometry, permute the dofmap
  if (coordinate_element.needs_permutation_data())
  {
    const int D = topology.dim();
    const int num_cells = topology.connectivity(D, 0)->num_nodes();
    const std::vector<std::uint32_t>& cell_info
        = topology.get_cell_permutation_info();

    for (std::int32_t cell = 0; cell < num_cells; ++cell)
      coordinate_element.unpermute_dofs(dofmap.links(cell).data(),
                                        cell_info[cell]);
  }

  // Build list of unique (global) node indices from adjacency list
  // (geometry nodes)
  std::vector<std::int64_t> indices = cell_nodes.array();
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

  //  Fetch node coordinates by global index from other ranks. Order of
  //  coords matches order of the indices in 'indices'
  common::array2d<double> coords
      = graph::build::distribute_data<double>(comm, indices, x);

  // Compute local-to-global map from local indices in dofmap to the
  // corresponding global indices in cell_nodes
  std::vector l2g
      = graph::build::compute_local_to_global_links(cell_nodes, dofmap);

  // Compute local (dof) to local (position in coords) map from (i)
  // local-to-global for dofs and (ii) local-to-global for entries in
  // coords
  std::vector l2l = graph::build::compute_local_to_local(l2g, indices);

  // Build coordinate dof array
  common::array2d<double> xg(coords.shape[0], coords.shape[1]);

  // Allocate space for input global indices
  std::vector<std::int64_t> igi(indices.size());

  for (std::size_t i = 0; i < coords.shape[0]; ++i)
  {
    for (std::size_t j = 0; j < coords.shape[1]; ++j)
      xg(i, j) = coords(l2l[i], j);
    igi[i] = indices[l2l[i]];
  }

  // If the mesh has higher order geometry, permute the dofmap
  if (coordinate_element.needs_permutation_data())
  {
    const int D = topology.dim();
    const int num_cells = topology.connectivity(D, 0)->num_nodes();
    const std::vector<std::uint32_t>& cell_info
        = topology.get_cell_permutation_info();

    for (std::int32_t cell = 0; cell < num_cells; ++cell)
      coordinate_element.permute_dofs(dofmap.links(cell).data(),
                                      cell_info[cell]);
  }

  return Geometry(dof_index_map, std::move(dofmap), coordinate_element,
                  std::move(xg), std::move(igi));
}
//-----------------------------------------------------------------------------

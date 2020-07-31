// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Geometry.h"
#include "Partitioning.h"
#include <boost/functional/hash.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/DofMapBuilder.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/Partitioning.h>
#include <sstream>

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
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& Geometry::x()
{
  return _x;
}
//-----------------------------------------------------------------------------
const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>&
Geometry::x() const
{
  return _x;
}
//-----------------------------------------------------------------------------
const fem::CoordinateElement& Geometry::cmap() const { return _cmap; }
//-----------------------------------------------------------------------------
Eigen::Vector3d Geometry::node(int n) const
{
  return _x.row(n).matrix().transpose();
}
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>& Geometry::input_global_indices() const
{
  return _input_global_indices;
}
//-----------------------------------------------------------------------------
std::size_t Geometry::hash() const
{
  // Compute local hash
  boost::hash<std::vector<double>> dhash;

  std::vector<double> data(_x.data(), _x.data() + _x.size());
  const std::size_t local_hash = dhash(data);
  return local_hash;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
mesh::Geometry mesh::create_geometry(
    MPI_Comm comm, const Topology& topology,
    const fem::CoordinateElement& coordinate_element,
    const graph::AdjacencyList<std::int64_t>& cell_nodes,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& x)
{
  // TODO: make sure required entities are initialised, or extend
  // fem::DofMapBuilder::build to take connectivities

  //  Build 'geometry' dofmap on the topology
  auto [dof_index_map, dofmap] = fem::DofMapBuilder::build(
      comm, topology, coordinate_element.dof_layout());

  // Build list of unique (global) node indices from adjacency list
  // (geometry nodes)
  std::vector<std::int64_t> indices(cell_nodes.array().data(),
                                    cell_nodes.array().data()
                                        + cell_nodes.array().rows());
  std::sort(indices.begin(), indices.end());
  indices.erase(std::unique(indices.begin(), indices.end()), indices.end());

  //  Fetch node coordinates by global index from other ranks. Order of
  //  coords matches order of the indices in 'indices'
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coords
      = graph::Partitioning::distribute_data<double>(comm, indices, x);

  // Compute local-to-global map from local indices in dofmap to the
  // corresponding global indices in cell_nodes
  std::vector l2g
      = graph::Partitioning::compute_local_to_global_links(cell_nodes, dofmap);

  // Compute local (dof) to local (position in coords) map from (i)
  // local-to-global for dofs and (ii) local-to-global for entries in
  // coords
  std::vector l2l = graph::Partitioning::compute_local_to_local(l2g, indices);

  // Build coordinate dof array
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xg(
      coords.rows(), coords.cols());

  // Allocate space for input global indices
  std::vector<std::int64_t> igi(indices.size());

  for (int i = 0; i < coords.rows(); ++i)
  {
    xg.row(i) = coords.row(l2l[i]);
    igi[i] = indices[l2l[i]];
  }

  return Geometry(dof_index_map, std::move(dofmap), coordinate_element,
                  std::move(xg), std::move(igi));
}
//-----------------------------------------------------------------------------

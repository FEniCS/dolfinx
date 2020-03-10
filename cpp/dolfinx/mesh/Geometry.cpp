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
#include <sstream>

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
Geometry::Geometry(std::shared_ptr<const common::IndexMap> index_map,
                   const graph::AdjacencyList<std::int32_t>& dofmap,
                   const fem::ElementDofLayout& layout,
                   const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor>& x,
                   const std::vector<std::int64_t>& global_indices)
    : _dim(x.cols()), _dofmap(dofmap), _index_map(index_map), _layout(layout),
      _global_indices(global_indices)
{
  if (x.rows() != (int)global_indices.size())
    throw std::runtime_error("Size mis-match");

  // Make all geometry 3D
  if (_dim == 3)
    _x = x;
  else
  {
    _x.resize(x.rows(), 3);
    _x.setZero();
    _x.block(0, 0, x.rows(), x.cols()) = x;
  }
}
//-----------------------------------------------------------------------------
int Geometry::dim() const { return _dim; }
//-----------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t>& Geometry::dofmap() { return _dofmap; }
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
Eigen::Ref<const Eigen::Vector3d> Geometry::x(int n) const
{
  return _x.row(n).matrix().transpose();
}
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>& Geometry::global_indices() const
{
  return _global_indices;
}
//-----------------------------------------------------------------------------
const fem::ElementDofLayout& Geometry::dof_layout() const { return _layout; }
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
std::string Geometry::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    for (Eigen::Index i = 0; i < _x.rows(); i++)
    {
      s << "  " << i << ":";
      for (Eigen::Index d = 0; d < _x.cols(); d++)
        s << " " << _x(i, d);
      s << std::endl;
    }
    s << std::endl;
  }
  else
  {
    s << "<Geometry of dimension " << _x.cols() << " and size " << _x.rows()
      << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
mesh::Geometry mesh::create_geometry(
    MPI_Comm comm, const Topology& topology,
    const fem::ElementDofLayout& layout,
    const graph::AdjacencyList<std::int64_t>& cells,
    const graph::AdjacencyList<std::int32_t>& dest, const std::vector<int>& src,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& x)
{
  // TODO: make sure required entities are initialised, or extend
  // fem::DofMapBuilder::build to take connectivities

  //  Build 'geometry' dofmap on the topology
  auto [dof_index_map, dofmap]
      = fem::DofMapBuilder::build(comm, topology, layout, 1);

  // Send/receive the 'cell nodes' (includes high-order geometry
  // nodes), and the global input cell index.
  //
  //  NOTE: Maybe we can ensure that the 'global cells' are in the same
  //  order as the owned cells (maybe they are already) to avoid the
  //  need for global_index_nodes
  //
  //  NOTE: This could be optimised as we have earlier computed which
  //  processes own the cells this process needs.
  std::set<int> _src(src.begin(), src.end());
  auto [cell_nodes, global_index_cell]
      = Partitioning::exchange(comm, cells, dest, _src);

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
      = Partitioning::distribute_data(comm, indices, x);

  // Compute local-to-global map from local indices in dofmap to the
  // corresponding global indices in cell_nodes
  std::vector<std::int64_t> l2g
      = Partitioning::compute_local_to_global_links(cell_nodes, dofmap);

  // Compute local (dof) to local (position in coords) map from (i)
  // local-to-global for dofs and (ii) local-to-global for entries in
  // coords
  std::vector<std::int32_t> l2l
      = Partitioning::compute_local_to_local(l2g, indices);

  // Build coordinate dof array
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xg(
      coords.rows(), coords.cols());
  for (int i = 0; i < coords.rows(); ++i)
    xg.row(i) = coords.row(l2l[i]);

  return Geometry(dof_index_map, dofmap, layout, xg, l2g);
}
//-----------------------------------------------------------------------------

// Copyright (C) 2006-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Geometry.h"
#include <boost/functional/hash.hpp>
#include <dolfinx/common/IndexMap.h>
#include <sstream>

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
Geometry::Geometry(std::shared_ptr<const common::IndexMap> index_map,
                   const graph::AdjacencyList<std::int32_t>& dofmap,
                   const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor>& x,
                   const std::vector<std::int64_t>& global_indices, int degree)
    : _dim(x.cols()), _global_indices(global_indices), _index_map(index_map),
      _dofmap(dofmap), _degree(degree)
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
Geometry::Geometry(
    std::int64_t num_points_global,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        x,
    const std::vector<std::int64_t>& global_indices,
    const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>&
        coordinate_dofs,
    int degree)
    : _dim(x.cols()), _global_indices(global_indices),
      _num_points_global(num_points_global), _dofmap(coordinate_dofs),
      _degree(degree)
{
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
std::size_t Geometry::num_points_global() const
{
  if (_index_map)
    return _index_map->size_global();
  else
    return _num_points_global;
}
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>& Geometry::global_indices() const
{
  return _global_indices;
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
int Geometry::degree() const { return _degree; }
//-----------------------------------------------------------------------------

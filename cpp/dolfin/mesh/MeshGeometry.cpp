// Copyright (C) 2006 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshGeometry.h"
#include <boost/functional/hash.hpp>
#include <dolfin/geometry/Point.h>
#include <sstream>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
MeshGeometry::MeshGeometry(
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>
        points)
    : _coordinates(points)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t MeshGeometry::dim() const { return _coordinates.cols(); }

//-----------------------------------------------------------------------------
std::size_t MeshGeometry::num_points() const { return _coordinates.rows(); }

//-----------------------------------------------------------------------------
std::size_t MeshGeometry::num_points_global() const
{
  return _num_points_global;
}
//-----------------------------------------------------------------------------
Eigen::Ref<const Eigen::Array<double, 1, Eigen::Dynamic>>
MeshGeometry::x(std::size_t n) const
{
  return _coordinates.row(n);
}
//-----------------------------------------------------------------------------
geometry::Point MeshGeometry::point(std::size_t n) const
{
  return geometry::Point(_coordinates.cols(), _coordinates.row(n).data());
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
MeshGeometry::points()
{
  return _coordinates;
}
//-----------------------------------------------------------------------------
const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
MeshGeometry::points() const
{
  return _coordinates;
}
//-----------------------------------------------------------------------------
const std::vector<std::int64_t>& MeshGeometry::global_indices() const
{
  return _global_indices;
}
//-----------------------------------------------------------------------------
void MeshGeometry::init(
    std::uint64_t num_points_global,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        coordinates,
    const std::vector<std::int64_t>& global_indices)
{
  _num_points_global = num_points_global;
  _coordinates = coordinates;
  _global_indices = global_indices;
}
//-----------------------------------------------------------------------------
std::size_t MeshGeometry::hash() const
{
  // Compute local hash
  boost::hash<std::vector<double>> dhash;

  std::vector<double> _x(_coordinates.data(),
                         _coordinates.data() + _coordinates.size());
  const std::size_t local_hash = dhash(_x);
  return local_hash;
}
//-----------------------------------------------------------------------------
std::string MeshGeometry::str(bool verbose) const
{
  std::stringstream s;
  if (verbose)
  {
    s << str(false) << std::endl << std::endl;
    for (Eigen::Index i = 0; i < _coordinates.rows(); i++)
    {
      s << "  " << i << ":";
      for (Eigen::Index d = 0; d < _coordinates.cols(); d++)
        s << " " << _coordinates(i, d);
      s << std::endl;
    }
    s << std::endl;
  }
  else
  {
    s << "<MeshGeometry of dimension " << _coordinates.cols() << " and size "
      << _coordinates.rows() << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------

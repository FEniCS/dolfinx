// Copyright (C) 2006 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Geometry.h"
#include <boost/functional/hash.hpp>
#include <sstream>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
Geometry::Geometry(std::int64_t num_points_global,
                   const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor>& coordinates,
                   const std::vector<std::int64_t>& global_indices)
    : _dim(coordinates.cols()), _global_indices(global_indices),
      _num_points_global(num_points_global)
{
  // Make all geometry 3D
  if (_dim == 3)
    _coordinates = coordinates;
  else
  {
    _coordinates.resize(coordinates.rows(), 3);
    _coordinates.setZero();
    _coordinates.block(0, 0, coordinates.rows(), coordinates.cols())
        = coordinates;
  }
}
//-----------------------------------------------------------------------------
std::size_t Geometry::dim() const { return _dim; }
//-----------------------------------------------------------------------------
std::size_t Geometry::num_points() const { return _coordinates.rows(); }

//-----------------------------------------------------------------------------
std::size_t Geometry::num_points_global() const { return _num_points_global; }
//-----------------------------------------------------------------------------
Eigen::Ref<const Eigen::Vector3d> Geometry::x(std::size_t n) const
{
  return _coordinates.row(n).matrix().transpose();
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& Geometry::points()
{
  return _coordinates;
}
//-----------------------------------------------------------------------------
const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>&
Geometry::points() const
{
  return _coordinates;
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

  std::vector<double> _x(_coordinates.data(),
                         _coordinates.data() + _coordinates.size());
  const std::size_t local_hash = dhash(_x);
  return local_hash;
}
//-----------------------------------------------------------------------------
std::string Geometry::str(bool verbose) const
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
    s << "<Geometry of dimension " << _coordinates.cols() << " and size "
      << _coordinates.rows() << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------

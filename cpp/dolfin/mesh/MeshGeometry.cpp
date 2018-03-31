// Copyright (C) 2006 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshGeometry.h"
#include <boost/functional/hash.hpp>
#include <dolfin/log/log.h>
#include <sstream>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
MeshGeometry::MeshGeometry(const Eigen::Ref<const EigenRowArrayXXd>& points)
    : coordinates(points) //_dim(points.cols())
{
  // // Resize geometry
  // coordinates.resize(points.rows() * _dim);

  // // Map and copy data
  // Eigen::Map<EigenRowArrayXXd> _x(coordinates.data(), points.rows(), _dim);
  // _x = points;
}
//-----------------------------------------------------------------------------
geometry::Point MeshGeometry::point(std::size_t n) const
{
  return geometry::Point(coordinates.cols(), this->x(n));
}
//-----------------------------------------------------------------------------
// void MeshGeometry::set(std::size_t local_index, const double* x)
// {
//   std::copy(x, x + _dim, coordinates.begin() + local_index * _dim);
// }
//-----------------------------------------------------------------------------
std::size_t MeshGeometry::hash() const
{
  // Compute local hash
  boost::hash<std::vector<double>> dhash;

  std::vector<double> _x(coordinates.data(),
                         coordinates.data() + coordinates.size());
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
    for (std::size_t i = 0; i < num_vertices(); i++)
    {
      s << "  " << i << ":";
      for (Eigen::Index d = 0; d < coordinates.cols(); d++)
        s << " " << x(i, d);
      s << std::endl;
    }
    s << std::endl;
  }
  else
  {
    s << "<MeshGeometry of dimension " << coordinates.cols() << " and size "
      << num_vertices() << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------

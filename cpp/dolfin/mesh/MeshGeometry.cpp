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
    : _coordinates(points)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
geometry::Point MeshGeometry::point(std::size_t n) const
{
  return geometry::Point(_coordinates.cols(), _coordinates.row(n).data());
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

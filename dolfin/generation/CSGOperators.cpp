// Copyright (C) 2012 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2012-04-13
// Last changed: 2012-04-13

#include <sstream>

#include <dolfin/common/utils.h>
#include <dolfin/log/log.h>
#include "CSGOperators.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
// CSGUnion
//-----------------------------------------------------------------------------
CSGUnion::CSGUnion(boost::shared_ptr<CSGGeometry> g0,
                   boost::shared_ptr<CSGGeometry> g1)
  : _g0(g0), _g1(g1)
{
  assert(g0);
  assert(g1);

  // Check dimensions
  if (g0->dim() != g1->dim())
  {
    dolfin_error("CSGOperators.cpp",
                 "create union of CSG geometries",
                 "Dimensions of geomestries don't match (%d vs %d)",
                 g0->dim(), g1->dim());
  }
}
//-----------------------------------------------------------------------------
dolfin::uint CSGUnion::dim() const
{
  assert(_g0->dim() == _g1->dim());
  return _g0->dim();
}
//-----------------------------------------------------------------------------
std::string CSGUnion::str(bool verbose) const
{
  assert(_g0);
  assert(_g1);

  std::stringstream s;

  if (verbose)
  {
    s << "<Union>\n"
      << "{\n"
      << indent(_g0->str(true))
      << "\n"
      << indent(_g1->str(true))
      << "\n}";
  }
  else
  {
    s << "(" << _g0->str(false) << " + " << _g1->str(false) << ")";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
// CSGIntersection
//-----------------------------------------------------------------------------
CSGIntersection::CSGIntersection(boost::shared_ptr<CSGGeometry> g0,
                                 boost::shared_ptr<CSGGeometry> g1)
  : _g0(g0), _g1(g1)
{
  assert(g0);
  assert(g1);

  // Check dimensions
  if (g0->dim() != g1->dim())
  {
    dolfin_error("CSGOperators.cpp",
                 "create intersection of CSG geometries",
                 "Dimensions of geomestries don't match (%d vs %d)",
                 g0->dim(), g1->dim());
  }
}
//-----------------------------------------------------------------------------
dolfin::uint CSGIntersection::dim() const
{
  assert(_g0->dim() == _g1->dim());
  return _g0->dim();
}
//-----------------------------------------------------------------------------
std::string CSGIntersection::str(bool verbose) const
{
  assert(_g0);
  assert(_g1);

  std::stringstream s;

  if (verbose)
  {
    s << "<Intersection>\n"
      << "{\n"
      << indent(_g0->str(true))
      << "\n"
      << indent(_g1->str(true))
      << "\n}";
  }
  else
  {
    s << "(" << _g0->str(false) << " * " << _g1->str(false) << ")";
  }

  return s.str();
}
//-----------------------------------------------------------------------------

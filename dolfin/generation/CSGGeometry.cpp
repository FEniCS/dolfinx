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
// First added:  2012-04-11
// Last changed: 2012-04-13

#include "CSGGeometry.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CSGGeometry::CSGGeometry()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CSGGeometry::~CSGGeometry()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::size_t CSGGeometry::subdomain(boost::shared_ptr<CSGGeometry> s)
{
  dolfin_assert(dim() == s->dim());
  subdomains.push_back(s);
  return subdomains.size();
}

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
// Modified by Benjamin Kehlet, 2013
//
// First added:  2012-04-11
// Last changed: 2013-04-18

#include "CSGGeometry.h"
#include <dolfin/common/NoDeleter.h>

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
void CSGGeometry::set_subdomain(std::size_t i, boost::shared_ptr<CSGGeometry> s)
{
  dolfin_assert(dim() == s->dim());

  if (i == 0)
  {
    error("Setting reserved CSG subdomain (0)");
  }

  // Check if i already used
  std::list<std::pair<std::size_t, boost::shared_ptr<const CSGGeometry> > >::iterator it = subdomains.begin();
  while (it != subdomains.end())
  {
    if (it->first == i)
    {
       warning("Double declaration of CSG subdomain with index %u.", i);

       // Remove existing declaration
       std::list<std::pair<std::size_t, boost::shared_ptr<const CSGGeometry> > >::iterator tmp = it;
       it++;
       subdomains.erase(tmp);
    }
    else
      ++it;
  }

  subdomains.push_back(std::make_pair(i, s));
}
//-----------------------------------------------------------------------------
void CSGGeometry::set_subdomain(std::size_t i, CSGGeometry& s)
{
  set_subdomain(i, reference_to_no_delete_pointer(s));
}

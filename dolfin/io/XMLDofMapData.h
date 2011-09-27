// Copyright (C) 2011 Garth N. Wells
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
// First added:  2011-09-24
// Last changed:

#ifndef __XMLDOFMAP_H
#define __XMLDOFMAP_H

#include <map>
#include <ostream>
#include <vector>
#include <dolfin/common/types.h>

namespace pugi
{
  class xml_node;
}

namespace dolfin
{

  class DofMap;
  class GenericDofMap;

  class XMLDofMapData
  {
  public:

    // Read XML DofMap
    static void read(std::map<uint, std::vector<uint> >& dofmap,
                     const pugi::xml_node xml_dolfin);

    /// Write GenericDofMap data to XML file
    static void write(const GenericDofMap& dofmap, pugi::xml_node xml_node);

  };

}

#endif

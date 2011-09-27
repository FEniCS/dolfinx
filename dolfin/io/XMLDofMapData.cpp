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
// First added:  2002-12-06
// Last changed: 2011-06-30

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include "pugixml.hpp"

#include "dolfin/common/MPI.h"
#include "dolfin/fem/DofMap.h"
#include "dolfin/fem/GenericDofMap.h"
#include "XMLDofMapData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void XMLDofMapData::read(std::map<uint, std::vector<uint> >& dofmap,
                     const pugi::xml_node xml_dolfin)
{
  // Check that we have a XML dof map
  const pugi::xml_node xml_dofmap_node = xml_dolfin.child("dof_map");
  if (!xml_dofmap_node)
    error("XMLDofMap::read: not a DOLFIN dof map file.");

  // Get size
  //const unsigned int size = xml_dofmap_node.attribute("size").as_uint();

  // Iterate over each cell entry
  for (pugi::xml_node_iterator it = xml_dofmap_node.begin(); it != xml_dofmap_node.end(); ++it)
  {
    const uint cell_index = it->attribute("cell_index").as_uint();
    const uint local_size = it->attribute("local_size").as_uint();
    std::vector<dolfin::uint> cell_map(local_size);
    cell_map.resize(local_size);
    for (uint i = 0; i < local_size; ++i)
    {
      const std::string label = "dof" + boost::lexical_cast<std::string>(i);
      cell_map[i] = it->attribute(label.c_str()).as_uint();
    }
    dofmap.insert(std::make_pair(cell_index, cell_map));

  }
}
//-----------------------------------------------------------------------------
void XMLDofMapData::write(const GenericDofMap& dofmap, pugi::xml_node xml_node)
{
  // FIXME: Need to gather dof map from each process, and to use
  //        global cell indices
  not_working_in_parallel("XMLDofMap::write");

  // Add dofmap node
  pugi::xml_node array_node = xml_node.append_child("dof_map");

  // Add size attribute
  const unsigned int size = dofmap.global_dimension();
  array_node.append_attribute("size") = size;

  // Loop over dof map for each cell
  std::vector<dolfin::uint>::const_iterator dof;
  for (uint i = 0; i < size; ++i)
  {
    pugi::xml_node cell_node = array_node.append_child("cell_map");
    const std::vector<dolfin::uint>& cell_map = dofmap.cell_dofs(i);
    cell_node.append_attribute("cell_index") = i;
    cell_node.append_attribute("local_size") = (uint) cell_map.size();
    for (dof = cell_map.begin(); dof != cell_map.end(); ++dof)
    {
      const std::string label = "dof" + boost::lexical_cast<std::string>(std::distance(cell_map.begin(), dof));
      cell_node.append_attribute(label.c_str()) = *dof;
    }
  }
}
//-----------------------------------------------------------------------------

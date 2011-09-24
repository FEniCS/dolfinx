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
#include "dolfin/fem/GenericDofMap.h"
#include "XMLDofMap.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void XMLDofMap::read(GenericDofMap& dofmap, const pugi::xml_node xml_dolfin)
{
  error("XMLDofMap::read not implemented.");
}
//-----------------------------------------------------------------------------
void XMLDofMap::write(const GenericDofMap& dofmap, pugi::xml_node xml_node)
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
    cell_node.append_attribute("cell_index") = i;
    const std::vector<dolfin::uint>& cell_map = dofmap.cell_dofs(i);
    for (dof = cell_map.begin(); dof != cell_map.end(); ++dof)
    {
      const std::string label = "dof" + boost::lexical_cast<std::string>(std::distance(cell_map.begin(), dof));
      cell_node.append_attribute(label.c_str()) = *dof;
    }
  }
}
//-----------------------------------------------------------------------------

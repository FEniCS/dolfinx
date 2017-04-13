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
// First added:  2011-09-27
// Last changed: 2011-09-27

#ifndef __XMLFUNCTIONDATA_H
#define __XMLFUNCTIONDATA_H

#include <ostream>
#include <vector>

namespace pugi
{
  class xml_node;
}

namespace dolfin
{

  class Function;

  /// I/O for XML representation of Function

  class XMLFunctionData
  {
  public:

    /// Read the XML file with function data
    static void read(Function& u, pugi::xml_node xml_node);

    /// Write the XML file with function data
    static void write(const Function& u, pugi::xml_node xml_node);

  private:

    static void build_global_to_cell_dof(std::vector<std::vector<
     std::pair<dolfin::la_index, dolfin::la_index> > >& global_dof_to_cell_dof,
     const FunctionSpace& V);

    static void build_dof_map(std::vector<std::vector<dolfin::la_index>
                              >& global_dof_to_cell_dof,
                              const FunctionSpace& V);

  };

}

#endif

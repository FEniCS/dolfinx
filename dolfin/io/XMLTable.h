// Copyright (C) 2015 Jan Blechta
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

#ifndef __XML_TABLE_H
#define __XML_TABLE_H

#include <ostream>
#include <string>

namespace pugi
{
  class xml_node;
}

namespace dolfin
{

  class Table;

  /// Output of XML representation of DOLFIN Table

  class XMLTable
  {
  public:

    /// Write the XML file
    static void write(const Table& table, pugi::xml_node xml_node);

  };

}

#endif

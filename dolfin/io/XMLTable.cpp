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

#include "pugixml.hpp"
#include <dolfin/log/log.h>
#include <dolfin/log/Table.h>
#include "XMLTable.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void XMLTable::write(const Table& table, pugi::xml_node xml_node)
{
  // Temporaries
  auto it_double(table.dvalues.cbegin());
  auto it_string(table.values.cbegin());

  // Add table node
  pugi::xml_node table_node = xml_node.append_child("table");
  table_node.append_attribute("name") = table.name().c_str();

  // Loop over rows and columns
  for (std::size_t i = 0; i < table.rows.size(); ++i)
  {
    pugi::xml_node row_node = table_node.append_child("row");
    row_node.append_attribute("key") = table.rows[i].c_str();
    for (std::size_t j = 0; j < table.cols.size(); ++j)
    {
      pugi::xml_node col_node = row_node.append_child("col");
      col_node.append_attribute("key") = table.cols[j].c_str();
      const auto key = std::make_pair(table.rows[i], table.cols[j]);
      it_double = table.dvalues.find(key);
      if (it_double != table.dvalues.end())
      {
        col_node.append_attribute("type") = "double";
        col_node.append_attribute("value") = it_double->second;
      }
      else
      {
        it_string = table.values.find(key);
        if (it_string == table.values.end())
          dolfin_error("XMLTable.cpp",
                       "write XML output for table",
                       "Table is not rectangular, element(%u, %u) is missing",
                       i, j);
        col_node.append_attribute("type") = "string";
        col_node.append_attribute("value") = it_string->second.c_str();
      }
    }
  }
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void XMLTable::read(Table& table, pugi::xml_node xml_node)
{

  // Check that we have a XML Table
  const pugi::xml_node xml_table = xml_node.child("table");
  if (!xml_table)
  {
    dolfin_error("XMLTable.cpp",
                 "read Table from XML file",
                 "Not a DOLFIN Table XML file");
  }

  // Check that there is only one root XML Table 
  if (xml_table.first_child().next_sibling())
  {
    dolfin_error("XMLTable.cpp",
                 "read table from XML file",
                 "Two tables are defined in XML file");
  }

  // Rename table in accordance with file input
  const std::string name = xml_table.attribute("name").value();
  table.rename(name, "");

  // Iterate over rows
  for (auto i = xml_table.begin(); i != xml_table.end(); ++i)
  {

    // Get row name
    const std::string row = i->attribute("key").value();
    
    // Iterate over columns
    pugi::xml_node xml_row_node = *i; 
    for (auto j = xml_row_node.begin(); j != xml_row_node.end(); ++j)
    {
      // Get column name
      const std::string col = j->attribute("key").value();

      // Get type of value
      const std::string type = j->attribute("type").value();

      // Get value
      const pugi::xml_attribute value = j->attribute("value");

      // Set table entry
      if (type == "double")
        table.set(row, col, value.as_double());
      else if (type == "int")
        table.set(row, col, value.as_int());
      else
      {
        dolfin_error("XMLTable.cpp",
                     "read Table from XML file",
                     "Not supported type (\"%s\") of table entry (\"%s\", \"%s\")",
                     type.c_str(), row.c_str(), col.c_str());
      }
    }
  }
  
}

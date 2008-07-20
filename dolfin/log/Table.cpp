// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-19
// Last changed: 2008-07-19

#include <sstream>
#include <iomanip>

#include <dolfin/log/log.h>
#include "Table.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Table::Table(std::string title) : title(title)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Table::~Table()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
TableEntry Table::operator() (std::string row, std::string col)
{
  TableEntry entry(row, col, *this);
  return entry;
}
//-----------------------------------------------------------------------------
dolfin::real Table::get(std::string row, std::string col) const
{
  std::pair<std::string, std::string> key(row, col);
  std::map<std::pair<std::string, std::string>, real>::const_iterator it = values.find(key);
  if (it == values.end())
    error("Missing table value for entry (\"%s\", \"%s\").", row.c_str(), col.c_str());
  return it->second;
}
//-----------------------------------------------------------------------------
void Table::set(std::string row, std::string col, real value)
{
  // Add row
  if (row_set.find(row) == row_set.end())
  {
    rows.push_back(row);
    row_set.insert(row);
  }

  // Add column
  if (col_set.find(col) == col_set.end())
  {
    cols.push_back(col);
    col_set.insert(col);
  }

  // Store value
  std::pair<std::string, std::string> key(row, col);
  values[key] = value;
}
//-----------------------------------------------------------------------------
void Table::disp() const
{
  std::vector<std::vector<std::string> > formatted_values;
  std::vector<uint> col_sizes;
  
  // Format values and compute column sizes
  col_sizes.push_back(title.size());
  for (uint j = 0; j < cols.size(); j++)
    col_sizes.push_back(cols[j].size());
  for (uint i = 0; i < rows.size(); i++)
  {
    formatted_values.push_back(std::vector<std::string>());
    col_sizes[0] = std::max(col_sizes[0], rows[i].size());
    for (uint j = 0; j < cols.size(); j++)
    {
      std::stringstream value;
      value << std::setprecision(5) << get(rows[i], cols[j]);
      formatted_values[i].push_back(value.str());
      col_sizes[j + 1] = std::max(col_sizes[j + 1], value.str().size());
    }
  }
  uint row_size = 2*col_sizes.size() + 1;
  for (uint j = 0; j < col_sizes.size(); j++)
    row_size += col_sizes[j];

  // Write table
  std::stringstream output;
  output << title;
  for (uint k = 0; k < col_sizes[0] - title.size(); k++) output << " ";
  output << "  |";
  for (uint j = 0; j < cols.size(); j++)
  {
    for (uint k = 0; k < col_sizes[j + 1] - cols[j].size(); k++) output << " ";
    output << "  " << cols[j];
  }
  output << std::endl;
  for (uint k = 0; k < row_size; k++) output << "-";
  output << std::endl;
  for (uint i = 0; i < rows.size(); i++)
  {
    output << rows[i];
    for (uint k = 0; k < col_sizes[0] - rows[i].size(); k++) output << " ";
    output << "  |";
    for (uint j = 0; j < cols.size(); j++)
    {
      for (uint k = 0; k < col_sizes[j + 1] - formatted_values[i][j].size(); k++) output << " ";
      output << "  " << formatted_values[i][j];
    }
    output << std::endl;
  }

  message(output.str());
}
//-----------------------------------------------------------------------------
TableEntry::TableEntry(std::string row, std::string col, Table& table)
  : row(row), col(col), table(table)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
TableEntry::~TableEntry()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const TableEntry& TableEntry::operator= (dolfin::real value)
{
  table.set(row, col, value);
  return *this;
}
//-----------------------------------------------------------------------------
TableEntry::operator dolfin::real() const
{
  return table.get(row, col);
}
//-----------------------------------------------------------------------------

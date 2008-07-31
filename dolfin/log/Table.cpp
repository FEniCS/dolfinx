// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-19
// Last changed: 2008-07-31

#include <iostream>

#include <sstream>
#include <iomanip>

#include <dolfin/common/constants.h>
#include <dolfin/log/LogStream.h>
#include <dolfin/log/log.h>
#include "Table.h"

using namespace dolfin;

typedef std::vector<std::string>::const_iterator iterator;

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
Table Table::operator+ (const Table& table) const
{
  // Check table sizes
  if (rows.size() != table.rows.size() || cols.size() != table.cols.size())
    error("Dimension mismatch for addition of tables.");

  // Add tables
  Table t;
  for (iterator i = rows.begin(); i !=rows.end(); i++)
    for (iterator j = cols.begin(); j != cols.end(); j++)
      t.set(*i, *j, get(*i, *j) + table.get(*i, *j));

  return t;
}
//-----------------------------------------------------------------------------
Table Table::operator- (const Table& table) const
{
  // Check table sizes
  if (rows.size() != table.rows.size() || cols.size() != table.cols.size())
    error("Dimension mismatch for addition of tables.");

  // Add tables
  Table t;
  for (iterator i = rows.begin(); i !=rows.end(); i++)
    for (iterator j = cols.begin(); j != cols.end(); j++)
      t.set(*i, *j, get(*i, *j) - table.get(*i, *j));

  return t;
}
//-----------------------------------------------------------------------------
const Table& Table::operator= (const Table& table)
{
  // Assign everything but the title
  
  rows = table.rows;
  row_set = table.row_set;

  cols = table.cols;
  col_set = table.col_set;

  values = table.values;

  return *this;
}
//-----------------------------------------------------------------------------
void Table::disp(bool round_to_zero) const
{
  if (rows.size() == 0 || cols.size() == 0)
    return;

  std::vector<std::vector<std::string> > formatted_values;
  std::vector<uint> col_sizes;
  
  // Format values and compute column sizes
  col_sizes.push_back(title.size());
  for (uint j = 0; j < cols.size(); j++)
    col_sizes.push_back(cols[j].size());
  for (uint i = 0; i < rows.size(); i++)
  {
    formatted_values.push_back(std::vector<std::string>());
    col_sizes[0] = std::max(col_sizes[0], (dolfin::uint)(rows[i].size()));
    for (uint j = 0; j < cols.size(); j++)
    {
      real value = get(rows[i], cols[j]);
      if (round_to_zero && std::abs(value) < DOLFIN_EPS)
        value = 0.0;
      std::stringstream string_value;
      string_value << std::setprecision(5) << value;
      formatted_values[i].push_back(string_value.str());
      col_sizes[j + 1] = std::max(col_sizes[j + 1], (dolfin::uint)(string_value.str().size()));
    }
  }
  uint row_size = 2*col_sizes.size() + 1;
  for (uint j = 0; j < col_sizes.size(); j++)
    row_size += col_sizes[j];

  // Write table
  cout << title;
  for (uint k = 0; k < col_sizes[0] - title.size(); k++) cout << " ";
  cout << "  |";
  for (uint j = 0; j < cols.size(); j++)
  {
    for (uint k = 0; k < col_sizes[j + 1] - cols[j].size(); k++) cout << " ";
    cout << "  " << cols[j];
  }
  cout << endl;
  for (uint k = 0; k < row_size; k++) cout << "-";
  cout << endl;
  for (uint i = 0; i < rows.size(); i++)
  {
    cout << rows[i];
    for (uint k = 0; k < col_sizes[0] - rows[i].size(); k++) cout << " ";
    cout << "  |";
    for (uint j = 0; j < cols.size(); j++)
    {
      for (uint k = 0; k < col_sizes[j + 1] - formatted_values[i][j].size(); k++) cout << " ";
      cout << "  " << formatted_values[i][j];
    }
    cout << endl;
  }
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

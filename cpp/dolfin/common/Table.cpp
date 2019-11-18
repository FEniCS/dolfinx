// Copyright (C) 2008-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Table.h"
#include <cfloat>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace dolfin;

//-----------------------------------------------------------------------------
Table::Table(std::string title, bool right_justify)
    : name(title), _right_justify(right_justify)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
TableEntry Table::operator()(std::string row, std::string col)
{
  TableEntry entry(row, col, *this);
  return entry;
}
//-----------------------------------------------------------------------------
void Table::set(std::string row, std::string col, int value)
{
  std::stringstream s;
  s << value;
  set(row, col, s.str());
  dvalues[std::pair(row, col)] = static_cast<double>(value);
}
//-----------------------------------------------------------------------------
void Table::set(std::string row, std::string col, std::size_t value)
{
  std::stringstream s;
  s << value;
  set(row, col, s.str());
  dvalues[std::pair(row, col)] = static_cast<double>(value);
}
//-----------------------------------------------------------------------------
void Table::set(std::string row, std::string col, double value)
{
  if (std::abs(value) < DBL_EPSILON)
    value = 0.0;
  std::stringstream s;
  s << std::setprecision(5) << value;
  set(row, col, s.str());
  dvalues[std::pair(row, col)] = value;
}
//-----------------------------------------------------------------------------
void Table::set(std::string row, std::string col, std::string value)
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
std::string Table::get(std::string row, std::string col) const
{
  std::pair<std::string, std::string> key(row, col);
  auto it = values.find(key);
  if (it == values.end())
  {
    throw std::runtime_error("Missing table value for entry (\"" + row
                             + "\", \"" + col + "\")");
  }

  return it->second;
}
//-----------------------------------------------------------------------------
double Table::get_value(std::string row, std::string col) const
{
  std::pair<std::string, std::string> key(row, col);
  auto it = dvalues.find(key);
  if (it == dvalues.end())
  {
    throw std::runtime_error("Missing double value for entry (\"" + row
                             + "\", \"" + col + "\")");
  }

  return it->second;
}
//-----------------------------------------------------------------------------
Table Table::reduce(MPI_Comm comm, Table::Reduction reduction)
{
  std::string new_title;

  // Prepare reduction operation y := op(y, x)
  void (*op_impl)(double&, const double&) = nullptr;
  switch (reduction)
  {
  case Table::Reduction::average:
    new_title = "[MPI_AVG] ";
    op_impl = [](double& y, const double& x) { y += x; };
    break;
  case Table::Reduction::min:
    new_title = "[MPI_MIN] ";
    op_impl = [](double& y, const double& x) {
      if (x < y)
        y = x;
    };
    break;
  case Table::Reduction::max:
    new_title = "[MPI_MAX] ";
    op_impl = [](double& y, const double& x) {
      if (x > y)
        y = x;
    };
    break;
  default:
    throw std::runtime_error("Cannot perform reduction of Table. Requested "
                             "reduction not implemented");
  }
  new_title += name;

  // Handle trivial reduction
  if (MPI::size(comm) == 1)
  {
    Table table_all(*this);
    table_all.name = new_title;
    return table_all;
  }

  // Get keys, values into containers
  std::string keys;
  std::vector<double> values;
  keys.reserve(128 * dvalues.size());
  values.reserve(dvalues.size());
  for (const auto& it : dvalues)
  {
    keys += it.first.first + '\0' + it.first.second + '\0';
    values.push_back(it.second);
  }

  // Gather to rank zero
  std::vector<std::string> keys_all;
  std::vector<double> values_all;
  MPI::gather(comm, keys, keys_all, 0);
  MPI::gather(comm, values, values_all, 0);

  // Return empty table on rank > 0
  if (MPI::rank(comm) > 0)
    return Table(new_title);

  // Construct dvalues map from obtained data
  std::map<std::array<std::string, 2>, double> dvalues_all;
  std::map<std::array<std::string, 2>, double>::iterator it;
  std::array<std::string, 2> key;
  key[0].reserve(128);
  key[1].reserve(128);
  double* values_ptr = values_all.data();
  for (std::size_t i = 0; i < MPI::size(comm); ++i)
  {
    std::stringstream keys_stream(keys_all[i]);
    while (std::getline(keys_stream, key[0], '\0'),
           std::getline(keys_stream, key[1], '\0'))
    {
      it = dvalues_all.find(key);
      if (it != dvalues_all.end())
        op_impl(it->second, *(values_ptr++));
      else
        dvalues_all[key] = *(values_ptr++);
    }
  }
  assert(values_ptr == values_all.data() + values_all.size());

  // Weight by MPI size when averaging
  if (reduction == Table::Reduction::average)
  {
    const double w = 1.0 / static_cast<double>(MPI::size(comm));
    for (auto& it : dvalues_all)
      it.second *= w;
  }

  // Construct table to return
  Table table_all(new_title);
  for (const auto& it : dvalues_all)
    table_all(it.first[0], it.first[1]) = it.second;

  return table_all;
}
//-----------------------------------------------------------------------------
std::string Table::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    std::vector<std::vector<std::string>> tvalues;
    std::vector<std::size_t> col_sizes;

    // Format values and compute column sizes
    col_sizes.push_back(name.size());
    for (std::size_t j = 0; j < cols.size(); j++)
      col_sizes.push_back(cols[j].size());
    for (std::size_t i = 0; i < rows.size(); i++)
    {
      tvalues.push_back(std::vector<std::string>());
      col_sizes[0] = std::max(col_sizes[0], rows[i].size());
      for (std::size_t j = 0; j < cols.size(); j++)
      {
        std::string value = get(rows[i], cols[j]);
        tvalues[i].push_back(value);
        col_sizes[j + 1] = std::max(col_sizes[j + 1], value.size());
      }
    }
    std::size_t row_size = 2 * col_sizes.size() + 1;
    for (std::size_t j = 0; j < col_sizes.size(); j++)
      row_size += col_sizes[j];

    // Stay silent if no data
    if (tvalues.empty())
      return "";

    // Write table
    s << name;
    for (std::size_t k = 0; k < col_sizes[0] - name.size(); k++)
      s << " ";
    s << "  |";
    for (std::size_t j = 0; j < cols.size(); j++)
    {
      if (_right_justify)
      {
        for (std::size_t k = 0; k < col_sizes[j + 1] - cols[j].size(); k++)
          s << " ";
        s << "  " << cols[j];
      }
      else
      {
        s << "  " << cols[j];
        for (std::size_t k = 0; k < col_sizes[j + 1] - cols[j].size(); k++)
          s << " ";
      }
    }
    s << "\n";
    for (std::size_t k = 0; k < row_size; k++)
      s << "-";
    for (std::size_t i = 0; i < rows.size(); i++)
    {
      s << "\n";
      s << rows[i];
      for (std::size_t k = 0; k < col_sizes[0] - rows[i].size(); k++)
        s << " ";
      s << "  |";
      for (std::size_t j = 0; j < cols.size(); j++)
      {
        if (_right_justify)
        {
          for (std::size_t k = 0; k < col_sizes[j + 1] - tvalues[i][j].size();
               k++)
            s << " ";
          s << "  " << tvalues[i][j];
        }
        else
        {
          s << "  " << tvalues[i][j];
          for (std::size_t k = 0; k < col_sizes[j + 1] - tvalues[i][j].size();
               k++)
            s << " ";
        }
      }
    }
  }
  else
  {
    s << "<Table of size " << rows.size() << " x " << cols.size() << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
std::string Table::str_latex() const
{
  if (rows.empty() || cols.empty())
    return "Empty table";

  std::stringstream s;

  s << name << "\n";
  s << "\\begin{center}\n";
  s << "\\begin{tabular}{|l|";
  for (std::size_t j = 0; j < cols.size(); j++)
    s << "|c";
  s << "|}\n";
  s << "\\hline\n";
  s << "& ";
  for (std::size_t j = 0; j < cols.size(); j++)
  {
    if (j < cols.size() - 1)
      s << cols[j] << " & ";
    else
      s << cols[j] << " \\\\\n";
  }
  s << "\\hline\\hline\n";
  for (std::size_t i = 0; i < rows.size(); i++)
  {
    s << rows[i] << " & ";
    for (std::size_t j = 0; j < cols.size(); j++)
    {
      if (j < cols.size() - 1)
        s << get(rows[i], cols[j]) << " & ";
      else
        s << get(rows[i], cols[j]) << " \\\\\n";
    }
    s << "\\hline\n";
  }
  s << "\\end{tabular}\n";
  s << "\\end{center}\n";

  return s.str();
}
//-----------------------------------------------------------------------------
TableEntry::TableEntry(std::string row, std::string col, Table& table)
    : _row(row), _col(col), _table(table)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const TableEntry& TableEntry::operator=(std::size_t value)
{
  _table.set(_row, _col, value);
  return *this;
}
//-----------------------------------------------------------------------------
const TableEntry& TableEntry::operator=(int value)
{
  _table.set(_row, _col, value);
  return *this;
}
//-----------------------------------------------------------------------------
const TableEntry& TableEntry::operator=(double value)
{
  _table.set(_row, _col, value);
  return *this;
}
//-----------------------------------------------------------------------------
const TableEntry& TableEntry::operator=(std::string value)
{
  _table.set(_row, _col, value);
  return *this;
}
//-----------------------------------------------------------------------------
TableEntry::operator std::string() const { return _table.get(_row, _col); }
//-----------------------------------------------------------------------------

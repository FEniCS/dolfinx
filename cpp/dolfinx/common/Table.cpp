// Copyright (C) 2008-2011 Anders Logg
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Table.h"
#include <array>
#include <dolfinx/common/MPI.h>
#include <functional>
#include <iostream>
#include <map>
#include <variant>

namespace
{
std::string to_str(std::variant<std::string, int, double> value)
{
  if (std::holds_alternative<int>(value))
    return std::to_string(std::get<int>(value));
  else if (std::holds_alternative<double>(value))
    return std::to_string(std::get<double>(value));
  else if (std::holds_alternative<std::string>(value))
    return std::get<std::string>(value);
  else
    throw std::runtime_error("Variant incorrect");
}
} // namespace

using namespace dolfinx;

//-----------------------------------------------------------------------------
Table::Table(std::string title, bool right_justify)
    : name(title), _right_justify(right_justify)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void Table::set(std::string row, std::string col,
                std::variant<std::string, int, double> value)
{
  // Add row
  if (std::find(_rows.begin(), _rows.end(), row) == _rows.end())
    _rows.push_back(row);

  // Add column
  if (std::find(_cols.begin(), _cols.end(), col) == _cols.end())
    _cols.push_back(col);

  // Store value
  std::pair<std::string, std::string> key(row, col);
  _values[key] = value;
}
//-----------------------------------------------------------------------------
std::variant<std::string, int, double> Table::get(std::string row,
                                                  std::string col) const
{
  std::pair<std::string, std::string> key(row, col);
  auto it = _values.find(key);
  if (it == _values.end())
  {
    throw std::runtime_error("Missing table value for entry (\"" + row
                             + "\", \"" + col + "\")");
  }

  return it->second;
}
//-----------------------------------------------------------------------------
Table Table::reduce(MPI_Comm comm, Table::Reduction reduction) const
{
  std::string new_title;

  // Prepare reduction operation y := op(y, x)
  std::function<double(double, double)> op_impl;
  switch (reduction)
  {
  case Table::Reduction::average:
    new_title = "[MPI_AVG] ";
    op_impl = [](double y, double x) { return y + x; };
    break;
  case Table::Reduction::min:
    new_title = "[MPI_MIN] ";
    op_impl = [](double y, double x) { return std::min(y, x); };
    break;
  case Table::Reduction::max:
    new_title = "[MPI_MAX] ";
    op_impl = [](double y, double x) { return std::max(y, x); };
    break;
  default:
    throw std::runtime_error("Cannot perform reduction of Table. Requested "
                             "reduction not implemented");
  }
  new_title += name;

  const int mpi_size = dolfinx::MPI::size(comm);

  // Handle trivial reduction
  if (mpi_size == 1)
  {
    Table table_all(*this);
    table_all.name = new_title;
    return table_all;
  }

  // Get keys, values into containers for int doubles
  std::string keys;
  std::vector<double> values;
  for (const auto& it : _values)
  {
    if (const auto* const pval = std::get_if<double>(&it.second))
    {
      keys += it.first.first + '\0' + it.first.second + '\0';
      values.push_back(*pval);
    }
  }

  // Gather to rank zero

  // Get string data size on each process
  std::vector<int> pcounts(mpi_size), offsets(mpi_size + 1, 0);
  const int local_size_str = keys.size();
  int err = MPI_Gather(&local_size_str, 1, MPI_INT, pcounts.data(), 1, MPI_INT,
                       0, comm);
  dolfinx::MPI::check_error(comm, err);
  std::partial_sum(pcounts.begin(), pcounts.end(), offsets.begin() + 1);
  std::vector<char> out_str(offsets.back());
  err = MPI_Gatherv(keys.data(), keys.size(), MPI_CHAR, out_str.data(),
                    pcounts.data(), offsets.data(), MPI_CHAR, 0, comm);
  dolfinx::MPI::check_error(comm, err);

  // Rebuild string
  std::vector<std::string> keys_all(mpi_size);
  for (int p = 0; p < mpi_size; ++p)
  {
    keys_all[p] = std::string(out_str.begin() + offsets[p],
                              out_str.begin() + offsets[p + 1]);
  }

  // Get value data size on each process
  const int local_size = values.size();
  err = MPI_Gather(&local_size, 1, MPI_INT, pcounts.data(), 1, MPI_INT, 0,
                   comm);
  dolfinx::MPI::check_error(comm, err);
  std::partial_sum(pcounts.begin(), pcounts.end(), offsets.begin() + 1);

  std::vector<double> values_all(offsets.back());
  err = MPI_Gatherv(values.data(), values.size(), MPI_DOUBLE, values_all.data(),
                    pcounts.data(), offsets.data(), MPI_DOUBLE, 0, comm);
  dolfinx::MPI::check_error(comm, err);

  // Return empty table on rank > 0
  if (MPI::rank(comm) > 0)
    return Table(new_title);

  // Construct dvalues map from obtained data
  std::map<std::array<std::string, 2>, double> dvalues_all;
  std::array<std::string, 2> key;
  const double* values_ptr = values_all.data();
  for (int i = 0; i < dolfinx::MPI::size(comm); ++i)
  {
    std::stringstream keys_stream(keys_all[i]);
    while (std::getline(keys_stream, key[0], '\0'),
           std::getline(keys_stream, key[1], '\0'))
    {
      if (auto it = dvalues_all.find(key); it != dvalues_all.end())
        it->second = op_impl(it->second, *(values_ptr++));
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
  for (const auto& it : _values)
  {
    if (std::holds_alternative<int>(it.second))
      table_all.set(it.first.first, it.first.second, it.second);
  }
  // NB - the cast to std::variant should not be needed: needed by Intel
  // compiler.
  for (const auto& it : dvalues_all)
  {
    table_all.set(it.first[0], it.first[1],
                  std::variant<std::string, int, double>(it.second));
  }

  return table_all;
}
//-----------------------------------------------------------------------------
std::string Table::str() const
{
  std::stringstream s;
  std::vector<std::vector<std::string>> tvalues;
  std::vector<std::size_t> col_sizes;

  // Format values and compute column sizes
  col_sizes.push_back(name.size());
  for (std::size_t j = 0; j < _cols.size(); j++)
    col_sizes.push_back(_cols[j].size());
  for (std::size_t i = 0; i < _rows.size(); i++)
  {
    tvalues.emplace_back();
    col_sizes[0] = std::max(col_sizes[0], _rows[i].size());
    for (std::size_t j = 0; j < _cols.size(); j++)
    {
      const std::string value = to_str(get(_rows[i], _cols[j]));
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
  for (std::size_t j = 0; j < _cols.size(); j++)
  {
    if (_right_justify)
    {
      for (std::size_t k = 0; k < col_sizes[j + 1] - _cols[j].size(); k++)
        s << " ";
      s << "  " << _cols[j];
    }
    else
    {
      s << "  " << _cols[j];
      for (std::size_t k = 0; k < col_sizes[j + 1] - _cols[j].size(); k++)
        s << " ";
    }
  }
  s << "\n";
  for (std::size_t k = 0; k < row_size; k++)
    s << "-";
  for (std::size_t i = 0; i < _rows.size(); i++)
  {
    s << "\n";
    s << _rows[i];
    for (std::size_t k = 0; k < col_sizes[0] - _rows[i].size(); k++)
      s << " ";
    s << "  |";
    for (std::size_t j = 0; j < _cols.size(); j++)
    {
      if (_right_justify)
      {
        for (std::size_t k = 0; k < col_sizes[j + 1] - tvalues[i][j].size();
             k++)
        {
          s << " ";
        }
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

  return s.str();
}
//-----------------------------------------------------------------------------

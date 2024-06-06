// Copyright (C) 2008-2011 Anders Logg
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <map>
#include <mpi.h>
#include <string>
#include <variant>
#include <vector>

namespace dolfinx
{

/// @brief This class provides storage and pretty-printing for tables.
///
/// Example usage:
///
///   Table table("Timings");
///   table.set("Foo", "Assemble", 0.010);
///   table.set("Foo", "Solve", 0.020);
///   table.set("Bar", "Assemble", 0.011);
///   table.set("Bar", "Solve", 0.019);
class Table
{
public:
  /// Types of MPI reduction available for Table, to get the max, min or
  /// average values over an MPI_Comm
  enum class Reduction
  {
    average,
    max,
    min
  };

  /// Create empty table
  Table(std::string title = "", bool right_justify = true);

  /// Copy constructor
  Table(const Table& table) = default;

  /// Move constructor
  Table(Table&& table) = default;

  /// Destructor
  ~Table() = default;

  /// Assignment operator
  Table& operator=(const Table& table) = default;

  /// Move assignment
  Table& operator=(Table&& table) = default;

  /// Set table entry
  /// @param[in] row Row name
  /// @param[in] col Column name
  /// @param[in] value The value to set
  void set(std::string row, std::string col,
           std::variant<std::string, int, double> value);

  /// Get value of table entry
  /// @param[in] row Row name
  /// @param[in] col Column name
  /// @returns Returns the entry for requested row and columns
  std::variant<std::string, int, double> get(std::string row,
                                             std::string col) const;

  /// Do MPI reduction on Table
  /// @param[in] comm MPI communicator
  /// @param[in] reduction Type of reduction to perform
  /// @return Reduced Table
  Table reduce(MPI_Comm comm, Reduction reduction) const;

  /// Table name
  std::string name;

  /// Return string representation of the table
  std::string str() const;

private:
  // Row and column names
  std::vector<std::string> _rows, _cols;

  // Table entry values
  std::map<std::pair<std::string, std::string>,
           std::variant<std::string, int, double>>
      _values;

  // True if we should right-justify the table entries
  bool _right_justify;
};

} // namespace dolfinx

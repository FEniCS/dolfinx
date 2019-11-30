// Copyright (C) 2008-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/MPI.h>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace dolfin
{
class TableEntry;

/// This class provides storage and pretty-printing for tables.
/// Example usage:
///
///   Table table("Timings");
///
///   table("Eigen",  "Assemble") = 0.010;
///   table("Eigen",  "Solve")    = 0.020;
///   table("PETSc",  "Assemble") = 0.011;
///   table("PETSc",  "Solve")    = 0.019;
///
///   log::info(table);

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

  // Set table entry
  void set(std::string row, std::string col,
           std::variant<std::string, int, double> value);

  /// Get value of table entry
  std::string get(std::string row, std::string col) const;

  /// Do MPI reduction on Table
  /// @param[in] comm MPI Comm
  /// @param[in] reduction Type of reduction to perform
  /// @return Reduced Table
  Table reduce(MPI_Comm comm, Reduction reduction);

  /// Table name
  std::string name;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

  /// Return informal string representation for LaTeX
  std::string str_latex() const;

private:
  // Rows and columns
  std::vector<std::string> _rows, _cols;

  // Table values as strings
  std::map<std::pair<std::string, std::string>,
           std::variant<std::string, int, double>>
      _values;

  // Table values as doubles
  std::map<std::pair<std::string, std::string>, double> _dvalues;

  // True if we should right-justify the table entries
  bool _right_justify;
};

} // namespace dolfin

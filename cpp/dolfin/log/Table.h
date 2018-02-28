// Copyright (C) 2008-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/Variable.h>
#include <map>
#include <set>
#include <vector>

namespace dolfin
{
class MPI;
class XMLTable;
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
///   table("Tpetra", "Assemble") = 0.012;
///   table("Tpetra", "Solve")    = 0.018;
///
///   info(table);

class Table : public common::Variable
{
public:
  /// Create empty table
  Table(std::string title = "", bool right_justify = true);

  /// Destructor
  ~Table();

  /// Return table entry
  TableEntry operator()(std::string row, std::string col);

  /// Set value of table entry
  void set(std::string row, std::string col, int value);

  /// Set value of table entry
  void set(std::string row, std::string col, std::size_t value);

  /// Set value of table entry
  void set(std::string row, std::string col, double value);

  /// Set value of table entry
  void set(std::string row, std::string col, std::string value);

  /// Get value of table entry
  std::string get(std::string row, std::string col) const;

  /// Get value of table entry
  double get_value(std::string row, std::string col) const;

  /// Assignment operator
  const Table& operator=(const Table& table);

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

  /// Return informal string representation for LaTeX
  std::string str_latex() const;

private:
  // Rows
  std::vector<std::string> rows;
  std::set<std::string> row_set;

  // Columns
  std::vector<std::string> cols;
  std::set<std::string> col_set;

  // Table values as strings
  std::map<std::pair<std::string, std::string>, std::string> values;

  // Table values as doubles
  std::map<std::pair<std::string, std::string>, double> dvalues;

  // True if we should right-justify the table entries
  bool _right_justify;

  // Allow MPI::all_reduce accessing dvalues
  friend class MPI;

  // Allow XMLTable accessing data
  friend class XMLTable;
};

/// This class represents an entry in a Table

class TableEntry
{
public:
  /// Create table entry
  TableEntry(std::string row, std::string col, Table& table);

  /// Destructor
  ~TableEntry();

  /// Assign value to table entry
  const TableEntry& operator=(std::size_t value);

  /// Assign value to table entry
  const TableEntry& operator=(int value);

  /// Assign value to table entry
  const TableEntry& operator=(double value);

  /// Assign value to table entry
  const TableEntry& operator=(std::string value);

  /// Cast to entry value
  operator std::string() const;

private:
  // Row
  std::string _row;

  // Column
  std::string _col;

  // Table
  Table& _table;
};
}



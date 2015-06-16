// Copyright (C) 2008-2011 Anders Logg
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
// First added:  2008-07-19
// Last changed: 2011-10-07

#ifndef __TABLE_H
#define __TABLE_H

#include <map>
#include <set>
#include <vector>
#include <dolfin/common/Variable.h>

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

  class Table : public Variable
  {
  public:

    /// Create empty table
    Table(std::string title="", bool right_justify=true);

    /// Destructor
    ~Table();

    /// Return table entry
    TableEntry operator() (std::string row, std::string col);

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
    const Table& operator= (const Table& table);

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
    const TableEntry& operator= (std::size_t value);

    /// Assign value to table entry
    const TableEntry& operator= (int value);

    /// Assign value to table entry
    const TableEntry& operator= (double value);

    /// Assign value to table entry
    const TableEntry& operator= (std::string value);

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

#endif

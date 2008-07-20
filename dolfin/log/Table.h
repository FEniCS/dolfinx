// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-19
// Last changed: 2008-07-19

#ifndef __TABLE_H
#define __TABLE_H

#include <vector>
#include <set>
#include <map>

#include <dolfin/common/types.h>

namespace dolfin
{

  class TableEntry;

  /// This class provides storage and pretty-printing for tables.
  /// Example usage:
  ///
  ///   Table table("Timings");
  ///
  ///   table("uBLAS",  "Assemble") = 0.010;
  ///   table("uBLAS",  "Solve")    = 0.020;
  ///   table("PETSc",  "Assemble") = 0.011;
  ///   table("PETSc",  "Solve")    = 0.019;
  ///   table("Epetra", "Assemble") = 0.012;
  ///   table("Epetra", "Solve")    = 0.018;
  ///
  ///   table.disp();

  class Table
  {
  public:

    /// Create empty table
    Table(std::string title="");

    /// Destructor
    ~Table();

    /// Return table entry
    TableEntry operator() (std::string row, std::string col);

    /// Get value of table entry
    real get(std::string row, std::string col) const;

    /// Set value of table entry
    void set(std::string row, std::string col, real value);

    /// Display table
    void disp() const;

  private:

    // Table title
    std::string title;

    // Rows
    std::vector<std::string> rows;
    std::set<std::string> row_set;

    // Columns
    std::vector<std::string> cols;
    std::set<std::string> col_set;

    // Table values
    std::map<std::pair<std::string, std::string>, real> values;

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
    const TableEntry& operator= (real value);

    /// Cast to entry value
    operator real() const;

  private:

    // Row
    std::string row;

    // Column
    std::string col;

    // Table
    Table& table;

  };

}

#endif

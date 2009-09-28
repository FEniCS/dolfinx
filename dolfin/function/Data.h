// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-11
// Last changed: 2009-09-28

#ifndef __DATA_H
#define __DATA_H

#include <ufc.h>
#include <dolfin/mesh/Point.h>

namespace dolfin
{

  class Cell;

  /// This class holds data for function evaluation, including
  /// the coordinates x, the time t, and auxiliary data that a
  /// function may depend on.

  class Data
  {
  public:

    /// Constructor
    Data();

    // FIXME: Remove this constructor after removing UFCFunction

    /// Constructor
    Data(const Cell& cell, int facet);

    /// Destructor
    ~Data();

    /// Return current cell (if available)
    const Cell& cell() const;

    /// Return current UFC cell (if available)
    const ufc::cell& ufc_cell() const;

    /// Return current facet (if available)
    uint facet() const;

    /// Return current facet normal (if available)
    Point normal() const;

    /// Return geometric dimension of cell
    uint geometric_dimension() const;

    /// Check if we are on a facet
    bool on_facet() const;

    /// Check if data is valid
    bool is_valid() const;

    /// The coordinates
    const double* x;

  private:

    // Friends
    friend class Expression;

    // FIXME: Remove these
    friend class UFCFunction;
    friend class Function;

    /// Update cell data
    void update(const Cell& dolfin_cell, const ufc::cell ufc_cell, int local_facet);

    /// Invalidate cell data
    void invalidate();

    // The current cell (if any, otherwise 0)
    const Cell* _dolfin_cell;

    // The current UFC cell (if any, otherwise 0)
    const ufc::cell* _ufc_cell;

    // The current facet (if any, otherwise -1)
    int _facet;

  };

}

#endif

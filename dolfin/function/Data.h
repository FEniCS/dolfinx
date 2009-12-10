// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-11
// Last changed: 2009-10-11

#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include <ufc.h>
#include <dolfin/common/Array.h>
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

    /// The coordinates
    Array<const double> x;

    /// Set cell and facet data
    void set(const Cell& dolfin_cell, const ufc::cell& ufc_cell, int local_facet);

    /// Set UFC cell and coordinate
    void set(const ufc::cell& ufc_cell, const double* x);

    /// Clear all cell data
    void clear();

  private:

    // Friends
    friend class GenericFunction;

    // FIXME: Remove this
    friend class Function;

    // The current cell (if any, otherwise 0)
    const Cell* _dolfin_cell;

    // The current UFC cell (if any, otherwise 0)
    const ufc::cell* _ufc_cell;

    // The current facet (if any, otherwise -1)
    int _facet;

  };

}

#endif

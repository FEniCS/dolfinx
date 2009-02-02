// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-11-03
// Last changed: 2008-11-03

#ifndef __DATA_H
#define __DATA_H

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

    /// Constructor
    Data(const Cell& cell, int facet);

    /// Destructor
    ~Data();

    /// Return current cell (if available)
    const Cell& cell() const;

    /// Return current facet (if available)
    uint facet() const;

    /// Return current facet normal (if available)
    Point normal() const;

    /// Check if we are on a facet
    bool on_facet() const;

    /// The coordinates
    const double* x;

    /// The current time
    const double t;

  private:

    // The current cell (if any, otherwise 0)
    const Cell* _cell;

    // The current facet (if any, otherwise -1)
    int _facet;


  };

}

#endif

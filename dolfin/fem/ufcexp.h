// Copyright (C) 2009 Shawn W. Walker.
// Licensed under the GNU LGPL Version 2.1.
//
// This file is to be used as a "testing-ground" for future additions to
// ufc.h (see the UFC project for more info).  That means when using the
// functionality in here, one must use dynamic_cast to access the
// data structures created here.
//
// First added:  2009-04-30
// Last changed: 2009-04-30

#ifndef __UFCEXP_H
#define __UFCEXP_H

#include <ufc.h>

namespace ufcexp
{

  /// This class defines the data structure for a cell in a mesh.

  class cell : public ufc::cell
  {
  public:

    /// Constructor
    cell(): ufc::cell(), higher_order_coordinates(0) {}

    /// Destructor
    virtual ~cell() {}

    /// Array of coordinates for higher order vertices of the cell
    double** higher_order_coordinates;

  };

}

#endif

// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-12
// Last changed: 2007-04-13

#ifndef __PROJECTION_LIBRARY_H
#define __PROJECTION_LIBRARY_H

#include <string>
#include <ufc.h>
#include <dolfin/fem/Form.h>

namespace dolfin
{

  /// Library of pregenerated L2 projections

  class ProjectionLibrary
  {
  public:

    /// Create projection forms with given signature
    static Form* create_projection_a(const char* signature);
    static Form* create_projection_L(const char* signature,
				     Function& f);
  };

}

#endif

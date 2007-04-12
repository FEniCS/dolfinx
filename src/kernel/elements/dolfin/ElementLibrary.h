// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-04-12
// Last changed: 2007-04-12

#ifndef __ELEMENT_LIBRARY_H
#define __ELEMENT_LIBRARY_H

#include <ufc.h>

namespace dolfin
{

  /// Library of pregenerated finite elements and dof maps.

  class ElementLibrary
  {
  public:

    /// Create finite element with given signature
    static ufc::finite_element* create_finite_element(const char* signature);

    /// Create dof map with given signature
    static ufc::dof_map* create_dof_map(const char* signature);

  };

}

#endif

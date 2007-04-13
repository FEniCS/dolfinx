// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-04-12
// Last changed: 2007-04-13

#ifndef __ELEMENT_LIBRARY_H
#define __ELEMENT_LIBRARY_H

#include <string>
#include <ufc.h>

namespace dolfin
{

  /// Library of pregenerated finite elements and dof maps.

  class ElementLibrary
  {
  public:

    /// Create finite element with given signature
    static ufc::finite_element* create_finite_element(const char* signature);

    /// Create finite element with given signature
    static ufc::finite_element* create_finite_element(std::string signature)
    { return create_finite_element(signature.c_str()); }

    /// Create dof map with given signature
    static ufc::dof_map* create_dof_map(const char* signature);

    /// Create dof map with given signature
    static ufc::dof_map* create_dof_map(std::string signature)
    { return create_dof_map(signature.c_str()); }

  };

}

#endif

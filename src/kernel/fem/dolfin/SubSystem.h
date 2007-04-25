// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-04-24
// Last changed: 2007-04-24

#ifndef __SUB_SYSTEM_H
#define __SUB_SYSTEM_H

#include <ufc.h>
#include <dolfin/Array.h>

namespace dolfin
{

  /// This class implements functionality for extracting nested sub
  /// finite elements and sub dof maps of a given finite element or
  /// dof map.

  class SubSystem
  {
  public:

    /// Extract given sub finite element
    static const ufc::finite_element* extractFiniteElement
    (const ufc::finite_element& finite_element, Array<uint>& sub_system);

    /// Extract given sub dof map
    static const ufc::dof_map* extractDofMap
    (const ufc::dof_map& dof_map, Array<uint>& sub_system);
    
  };

}

#endif

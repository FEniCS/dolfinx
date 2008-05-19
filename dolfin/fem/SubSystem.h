// Copyright (C) 2007-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2007-04-24
// Last changed: 2008-05-18

#ifndef __SUB_SYSTEM_H
#define __SUB_SYSTEM_H

#include <ufc.h>
#include <dolfin/common/Array.h>

namespace dolfin
{

  class Mesh;

  /// This class represents a sub system that may be specified as a
  /// recursively nested sub system of some given system.
  ///
  /// The sub system is specified by an array of indices. For example,
  /// the array [3, 0, 2] specifies sub system 2 of sub system 0 of
  /// sub system 3.

  class SubSystem
  {
  public:

    /// Create empty sub system (no sub systems)
    SubSystem();

    /// Create given sub system (one level)
    SubSystem(uint sub_system);

    /// Create given sub sub system (two levels)
    SubSystem(uint sub_system, uint sub_sub_system);

    /// Create sub system for given array (n levels)
    SubSystem(const Array<uint>& sub_system);

    /// Copy constructor
    SubSystem(const SubSystem& sub_system);

    /// Assignment operator
    const SubSystem& operator= (const SubSystem& sub_system);

    /// Return number of levels for nested sub system
    uint depth() const;
    
    /// Extract sub finite element of given finite element
    ufc::finite_element* extractFiniteElement
    (const ufc::finite_element& finite_element) const;

    /// Return array which defines sub system
    const Array<uint>& array() const
    { return sub_system; }

  private:

    // Recursively extract sub finite element
    static ufc::finite_element* extractFiniteElement
    (const ufc::finite_element& finite_element, const Array<uint>& sub_system);

    // The array specifying the sub system
    Array<uint> sub_system;
    
  };

}

#endif

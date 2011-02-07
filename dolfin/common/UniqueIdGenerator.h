// Copyright (C) 2010 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-12-05
// Last changed:

#ifndef __UNIQUE_ID_GENERATOR_H
#define __UNIQUE_ID_GENERATOR_H

#include <dolfin/common/types.h>

namespace dolfin
{

  // FIXME: Make a base class that classes can inherit from

  /// This is a singleton class that return IDs that are unique in the
  /// lifetime of a program.

  class UniqueIdGenerator
  {
  public:

    UniqueIdGenerator();

    /// Generate a unique ID
    static uint id();

  private:

    // Singleton instance
    static UniqueIdGenerator unique_id_generator;

    // Next ID to be returned
    uint next_id;

  };

}

#endif

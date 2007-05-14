// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-12-19
// Last changed: 2006-02-08

#ifndef __PARAMETER_SYSTEM_H
#define __PARAMETER_SYSTEM_H

#include <dolfin/ParameterList.h>

namespace dolfin
{

  /// This class holds a global database of parameters for DOLFIN,
  /// implemented as a set of (key, value) pairs. Supported value
  /// types are real, int, bool, and string.

  class ParameterSystem : public ParameterList
  {
  public:

    /// Singleton instance of global parameter database
    static ParameterSystem parameters;

  private:

    // Constructor
    ParameterSystem();
    
  };

}

#endif

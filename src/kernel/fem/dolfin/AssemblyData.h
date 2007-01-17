// Copyright (C) 2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2007-01-17
// Last changed: 2007-01-17

#ifndef __ASSEMBLY_DATA_H
#define __ASSEMBLY_DATA_H

#include <ufc.h>

namespace dolfin
{

  /// This class is a simple data structure that holds local data used
  /// during assembly.

  class AssemblyData
  {
  public:

    /// Constructor
    AssemblyData(const ufc::form& form);

    /// Destructor
    ~AssemblyData();

    

  };

}

#endif

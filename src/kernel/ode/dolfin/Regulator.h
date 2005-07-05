// Copyright (C) 2003-2005 Johan Jansson and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-11-13
// Last changed: 2005

#ifndef __REGULATOR_H
#define __REGULATOR_H

#include <dolfin/constants.h>

namespace dolfin
{

  /// Time step regulator.

  class Regulator
  {
  public:
    
    /// Constructor
    Regulator();

    /// Desctructor
    ~Regulator();

    /// Regulate time step
    real regulate(real knew, real k0, real kmax, bool kfixed);

  private:

    /// Time step conservation
    real w;

  };

}

#endif

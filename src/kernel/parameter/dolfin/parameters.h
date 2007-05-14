// Copyright (C) 2005-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2005-12-19
// Last changed: 2007-05-14

#ifndef __PARAMETERS_H
#define __PARAMETERS_H

#include <dolfin/Parameter.h>

namespace dolfin
{

  /// Add parameter
  void add(std::string key, dolfin::Parameter value);
  
  /// Set value of parameter
  void set(std::string key, dolfin::Parameter value);
  
  /// Get value of parameter with given key
  Parameter get(std::string key);

}

#endif

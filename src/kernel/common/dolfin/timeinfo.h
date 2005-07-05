// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-02-06
// Last changed: 2005

#ifndef __TIMEINFO_H
#define __TIMEINFO_H

#include <dolfin/constants.h>

namespace dolfin {

  /// Return a string containing current date and time
  const char* date();

  /// Start timing
  void tic();

  /// Return elapsed time
  real toc();

  /// Return and display elapsed time
  real tocd();

}

#endif

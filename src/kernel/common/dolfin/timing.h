// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-12-21
// Last changed: 2005-12-21

#ifndef __TIMING_H
#define __TIMING_H

#include <dolfin/constants.h>

namespace dolfin
{

  /// Start timing
  void tic();

  /// Return elapsed time
  real toc();

  /// Return and display elapsed time
  real tocd();

}

#endif

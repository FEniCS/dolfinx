// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-12-21
// Last changed: 2008-03-06

#ifndef __TIMING_H
#define __TIMING_H

#include <dolfin/common/types.h>

namespace dolfin
{

  /// Start timing
  void tic();

  /// Return elapsed CPU time
  real toc();

  /// Return and display elapsed CPU time
  real tocd();

  /// Return current CPU time used by process
  real time();

}

#endif

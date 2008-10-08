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
 
  /// Timing functions measure CPU time as determined by clock(),
  /// the precision of which seems to be 0.01 seconds.

  /// Start timing
  void tic();

  /// Return elapsed CPU time
  double toc();

  /// Return and display elapsed CPU time
  double tocd();

  /// Return current CPU time used by process
  double time();

}

#endif

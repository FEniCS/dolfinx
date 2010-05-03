// Copyright (C) 2005-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-12-21
// Last changed: 2010-05-03

#ifndef __TIMING_H
#define __TIMING_H

#include <dolfin/common/types.h>

namespace dolfin
{

  /// Timing functions measure CPU time as determined by clock(),
  /// the precision of which seems to be 0.01 seconds.

  /// Start timing (should not be used internally in DOLFIN!)
  void tic();

  /// Return elapsed CPU time (should not be used internally in DOLFIN!)
  double toc();

  /// Return current CPU time used by process
  double time();

}

#endif

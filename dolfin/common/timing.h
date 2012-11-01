// Copyright (C) 2005-2010 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2005-12-21
// Last changed: 2012-11-01

#ifndef __TIMING_H
#define __TIMING_H

#include <dolfin/common/types.h>
#include <dolfin/log/Table.h>

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

  /// Return a summary of timings and tasks in a Table, optionally clearing
  /// stored timings
  Table timings(bool reset=false);

  /// List a summary of timings and tasks, optionally clearing stored timings
  void list_timings(bool reset=false);

  /// This function is deprecated, use list_timings
  void summary(bool reset=false);

  /// Return timing (average) for given task, optionally clearing timing
  /// for task
  double timing(std::string task, bool reset=false);

}

#endif

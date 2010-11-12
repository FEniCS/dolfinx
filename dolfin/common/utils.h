// Copyright (C) 2009-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-08-09
// Last changed: 2010-11-12

#ifndef __UTILS_H
#define __UTILS_H

#include <string>
#include "types.h"

namespace dolfin
{

  /// Indent string block
  std::string indent(std::string block);

  /// Return string representation of int
  std::string to_string(int n);

  /// Return string representation of double
  std::string to_string(double x);

  /// Return string representation of given array
  std::string to_string(const double* x, uint n);

  /// Return simple hash for given signature string
  dolfin::uint hash(std::string signature);

}

#endif

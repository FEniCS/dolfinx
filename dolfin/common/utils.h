// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-08-09
// Last changed: 2009-10-06

#ifndef __UTILS_H
#define __UTILS_H

#include <string>
#include "types.h"

namespace dolfin
{

  /// Indent string block
  std::string indent(std::string block);

  /// Return simple hash for given signature string
  dolfin::uint hash(std::string signature);

}

#endif

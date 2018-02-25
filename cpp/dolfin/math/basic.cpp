// Copyright (C) 2003-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "basic.h"
#include <cmath>
#include <cstdlib>
#include <dolfin/common/constants.h>
#include <dolfin/log/log.h>
#include <time.h>

using namespace dolfin;

namespace dolfin
{
/// Flag to determine whether to reseed dolfin::rand().
/// Normally on first call.
bool rand_seeded = false;
}

//-----------------------------------------------------------------------------
std::size_t dolfin::ipow(std::size_t a, std::size_t n)
{
  // Treat special case a==0
  if (a == 0)
  {
    if (n == 0)
    {
      // size_t does not have NaN, raise error
      dolfin_error("math/basic.cpp", "take power ipow(0, 0)",
                   "ipow(0, 0) does not have a sense");
    }
    return 0;
  }

  std::size_t p = 1;
  // NOTE: Overflow not checked! Use __builtin_mul_overflow when
  //       GCC >= 5, Clang >= 3.8 is required!
  for (std::size_t i = 0; i < n; i++)
    p *= a;
  return p;
}
//-----------------------------------------------------------------------------

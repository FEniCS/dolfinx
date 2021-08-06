// Copyright (C) 2003-2011 Anders Logg
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "math.h"
#include <array>
#include <xtl/xspan.hpp>

extern "C"
{
  void dgesvd(char* jobu, char* jobvt, int* m, int* n, double* a, int* lda,
              double* s, double* u, int* ldu, double* vt, int* ldvt,
              double* work, int* lwork, int* info);
}

using namespace dolfinx;

//-----------------------------------------------------------------------
void math::pinv(xtl::span<const double> /* A */, std::array<int, 2> /* shape */,
                const xtl::span<double> /* P */)
{
  throw std::runtime_error("No implemented");
}
//-----------------------------------------------------------------------------

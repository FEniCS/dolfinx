// Copyright (C) 2003-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstddef>
#include <dolfin/common/constants.h>
#include <utility>

namespace dolfin
{
/// Return a to the power n.
/// NOTE: Overflow is not checked!
/// @param a (std::size_t)
///   Value
/// @param n (std::size_t)
///   Power
/// @return std::size_t
std::size_t ipow(std::size_t a, std::size_t n);

/// Check whether x is close to x0 (to within DOLFIN_EPS)
/// @param x (double)
///    First value
/// @param x0 (double)
///    Second value
/// @param eps (double)
///   Tolerance
/// @return bool
bool near(double x, double x0, double eps = DOLFIN_EPS);
}



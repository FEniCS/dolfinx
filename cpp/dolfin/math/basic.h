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
}

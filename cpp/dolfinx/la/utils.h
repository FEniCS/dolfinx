// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

namespace dolfinx::la
{

/// Norm types
enum class Norm
{
  l1,
  l2,
  linf,
  frobenius
};

} // namespace dolfinx::la
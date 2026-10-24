// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/mesh/ElementDofLayout.h>

namespace dolfinx::fem
{
// Deprecated: use mesh::ElementDofLayout. Hidden from Doxygen, which (in
// 1.15) misparses an attribute on an alias declaration.
/// @cond
using ElementDofLayout [[deprecated("Use mesh::ElementDofLayout instead.")]]
= mesh::ElementDofLayout;
/// @endcond
} // namespace dolfinx::fem

// Copyright (C) 2005-2011 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

namespace dolfinx
{

/// Initialize DOLFINX (and PETSc) with command-line arguments. This
/// should not be needed in most cases since the initialization is
/// otherwise handled automatically.
void init(int argc, char* argv[]);
} // namespace dolfinx

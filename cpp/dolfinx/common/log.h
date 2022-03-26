// Copyright (C) 2019 Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#define LOGURU_WITH_STREAMS 1
#define LOGURU_REPLACE_GLOG 1

#include "loguru.hpp"

namespace dolfinx
{

/// Initialise loguru
void init_logging(int argc, char* argv[]);

} // namespace dolfinx
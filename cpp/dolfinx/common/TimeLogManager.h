// Copyright (C) 2003-2016 Anders Logg
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "TimeLogger.h"

namespace dolfinx::common
{

/// Logger initialisation

class TimeLogManager
{
public:
  /// Singleton instance of logger
  static TimeLogger& logger();
};
} // namespace dolfinx::common

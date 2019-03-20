// Copyright (C) 2003-2016 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Logger.h"

namespace dolfin
{

namespace log
{

/// Logger initialisation

class LogManager
{
public:
  /// Singleton instance of logger
  static Logger& logger();
};
}
}


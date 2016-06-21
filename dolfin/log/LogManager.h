// Copyright (C) 2003-2016 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Thanks to Jim Tilander for many helpful hints.
//
// First added:  2003-03-13
// Last changed: 2016-06-10

#ifndef __LOG_MANAGER_H
#define __LOG_MANAGER_H

#include "Logger.h"

namespace dolfin
{

  class LogManager
  {
  public:

    // Singleton instance of logger
    static Logger& logger();

  };

}

#endif

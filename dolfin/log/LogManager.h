// Copyright (C) 2003-2007 Anders Logg
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
// Last changed: 2007-05-11

#ifndef __LOG_MANAGER_H
#define __LOG_MANAGER_H

#include "Logger.h"

namespace dolfin
{

  /* FIXME: logging in destructors may fail at exit because the logger instance
     may already be destroyed ("static initialization order fiasco"). The same
     may happen at startup, if logging from constructors of static objects. The
     logger instance should be converted to a heap-allocated (and never
     deleted) object, with an accessor function. Like in SubSystemsManager, but
     maybe with a static pointer rather than a static object in the accessor
     function. */

  class LogManager
  {
  public:

    // Singleton instance of logger
    static Logger logger;

  };

}

#endif

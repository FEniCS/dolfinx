// Copyright (C) 2003-2005 Anders Logg
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
// First added:  2003-03-13
// Last changed: 2005

#include "LogManager.h"

// Initialise static data
// FIXME : Logger singleton is initialised here on the first call to logger()
// to avoid "static initialisazation order fiasco". Logger's destructor
// may therefore never be called.

dolfin::Logger& dolfin::LogManager::logger()
{
  // NB static - this only allocates a new Logger on the first call to logger()
  static dolfin::Logger* lg = new(dolfin::Logger);
  return *lg;
}

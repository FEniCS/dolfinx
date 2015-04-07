// Copyright (C) 2005-2011 Anders Logg
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
// First added:  2005-02-13
// Last changed: 2011-03-17

#include <dolfin/common/constants.h>
#include <dolfin/log/log.h>
#include "SubSystemsManager.h"
#include "init.h"

//-----------------------------------------------------------------------------
void dolfin::init(int argc, char* argv[])
{
  log(PROGRESS, "Initializing DOLFIN version %s.", DOLFIN_VERSION);

  #ifdef HAS_PETSC
  SubSystemsManager::init_petsc(argc, argv);
  #endif
}
//-----------------------------------------------------------------------------

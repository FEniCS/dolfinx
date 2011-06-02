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
// Last changed: 2011-01-24

#ifndef __INIT_H
#define __INIT_H

namespace dolfin
{

  /// Initialize DOLFIN (and PETSc) with command-line arguments. This
  /// should not be needed in most cases since the initialization is
  /// otherwise handled automatically.
  void init(int argc, char* argv[]);

}

#endif

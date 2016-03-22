// Copyright (C) 2008 Solveig Bruvoll and Anders Logg
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
// First added:  2008-05-02
// Last changed: 2008-08-11

#ifndef __ALE_TYPE_H
#define __ALE_TYPE_H

namespace dolfin
{

  /// List of available methods for ALE mesh movement
  enum class ALEType {lagrange, hermite, harmonic};

}

#endif

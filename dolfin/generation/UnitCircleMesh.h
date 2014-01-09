// Copyright (C) 2005-2006 Anders Logg
// AL: I don't think I wrote this file, who did?
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
// Modified by Nuno Lopes 2008
// Modified by Anders Logg 2012
// Modified by Benjamin Kehlet 2012
//
// First added:  2005-12-02
// Last changed: 2012-11-09

#ifndef __UNIT_CIRCLE_MESH_H
#define __UNIT_CIRCLE_MESH_H

#include <string>
#include <dolfin/generation/EllipseMesh.h>

namespace dolfin
{

  /// This class is deprecated.

  class UnitCircleMesh : public CircleMesh
  {
  public:

    /// This class is deprecated
    UnitCircleMesh(std::size_t n,
                   std::string diagonal="crossed",
                   std::string transformation="rotsumn");

  };

}

#endif

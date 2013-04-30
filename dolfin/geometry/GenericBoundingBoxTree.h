// Copyright (C) 2013 Anders Logg
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
// First added:  2013-04-23
// Last changed: 2013-05-01

#ifndef __GENERIC_BOUNDING_BOX_TREE_H
#define __GENERIC_BOUNDING_BOX_TREE_H

namespace dolfin
{

  /// Base class for bounding box implementations

  class GenericBoundingBoxTree
  {
  public:

    /// Destructor
    virtual ~GenericBoundingBoxTree() {}

    /// Find entities intersecting the given _Point_
    virtual std::vector<unsigned int> find(const Point& point) const = 0;

  };

}

#endif

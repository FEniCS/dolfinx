// Copyright (C) 2010 Anders Logg
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
// First added:  2010-11-17
// Last changed: 2011-08-24

#ifndef __SUBSET_ITERATOR_H
#define __SUBSET_ITERATOR_H

#include <vector>

#include <dolfin/common/types.h>
#include <dolfin/log/log.h>
#include "MeshFunction.h"
#include "Mesh.h"
#include "MeshEntity.h"

namespace dolfin
{

  /// A _SubsetIterator_ is similar to a _MeshEntityIterator_ but
  /// iterates over a specified subset of the range of entities as
  /// specified by a _MeshFunction_ that labels the entites.

  class SubsetIterator
  {
  public:

    /// Create iterator for given mesh function. The iterator visits
    /// all entities that match the given label.
    SubsetIterator(const MeshFunction<unsigned int>& labels, uint label)
      : entity(labels.mesh(), labels.dim(), 0)
    {
      // Extract subset
      subset.clear();
      for (MeshEntityIterator entity(labels.mesh(), labels.dim()); !entity.end(); ++entity)
      {
        if (labels[*entity] == label)
          subset.push_back(entity->index());
      }

      // Set iterator
      it = subset.begin();
    }

    /// Destructor
    virtual ~SubsetIterator() {}

    /// Step to next mesh entity (prefix increment)
    SubsetIterator& operator++()
    {
      ++it;
      return *this;
    }

    /// Dereference operator
    MeshEntity& operator*()
    { return *operator->(); }

    /// Member access operator
    MeshEntity* operator->()
    { entity._index = *it; return &entity; }

    /// Check if iterator has reached the end
    bool end() const
    { return it == subset.end(); }

  private:

    // Mesh entity
    MeshEntity entity;

    // Subset
    std::vector<uint> subset;

    // Iterator
    std::vector<uint>::iterator it;

  };

}

#endif

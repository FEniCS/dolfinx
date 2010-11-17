// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2010-11-17
// Last changed: 2010-11-17

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
    SubsetIterator(const MeshFunction<uint>& labels, uint label)
      : entity(labels.mesh(), labels.dim(), 0)
    {
      // Extract subset
      subset.clear();
      for (MeshEntityIterator entity(labels.mesh(), labels.dim()); !entity.end(); ++entity)
      {
        if (labels[*entity] == label)
          subset.push_back(entity->index());
      }
      info("Iterating over subset, found %d entities out of %d.",
           subset.size(), labels.size());

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

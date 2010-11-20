// Copyright (C) 2006-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2009.
//
// First added:  2006-06-02
// Last changed: 2010-08-16

#ifndef __FACET_H
#define __FACET_H

#include "Cell.h"
#include "MeshEntity.h"
#include "MeshEntityIterator.h"
#include "MeshFunction.h"

namespace dolfin
{

  /// A Facet is a MeshEntity of topological codimension 1.

  class Facet : public MeshEntity
  {
  public:

    /// Constructor
    Facet(const Mesh& mesh, uint index) : MeshEntity(mesh, mesh.topology().dim() - 1, index) {}

    /// Destructor
    ~Facet() {}

    // FIXME: This function should take care of facet 'ownership' when a mesh
    //        is distributed across processes
    /// Determine whether or not facet is an interior facet. This is 'relative'
    /// to the given partition of the mesh if the mesh is distributed
    bool interior() const
    {
      not_working_in_parallel("Getting adjacent cell");

      if (num_entities(dim() + 1) == 2)
        return true;
      else
        return false;
    }

    // FIXME: This function should take care of the case where adjacent cells
    // FIXME: live on different processes

    /// Return adjacent cells. An optional argument that lists for
    /// each facet the index of the first cell may be given to specify
    /// the ordering of the two cells. If not specified, the ordering
    /// will depend on the (arbitrary) ordering of the mesh
    /// connectivity.
    std::pair<const Cell, const Cell> adjacent_cells(const MeshFunction<uint>* facet_orientation=0) const
    {
       assert(num_entities(dim() + 1) == 2);

       // Get cell indices
       const uint D = dim() + 1;
       const uint c0 = entities(D)[0];
       const uint c1 = entities(D)[1];

       // Normal ordering
       if (!facet_orientation || (*facet_orientation)[*this] == c0)
         return std::make_pair(Cell(mesh(), c0), Cell(mesh(), c1));

       // Sanity check
       if ((*facet_orientation)[*this] != c1)
       {
        error("Illegal facet orientation specified, cell %d is not a neighbor of facet %d.",
               (*facet_orientation)[*this], index());
        }

       // Opposite ordering
       return std::make_pair(Cell(mesh(), c1), Cell(mesh(), c0));
    }

  };

  /// A FacetIterator is a MeshEntityIterator of topological codimension 1.

  class FacetIterator : public MeshEntityIterator
  {
  public:

    FacetIterator(const Mesh& mesh) : MeshEntityIterator(mesh, mesh.topology().dim() - 1) {}
    FacetIterator(const MeshEntity& entity) : MeshEntityIterator(entity, entity.mesh().topology().dim() - 1) {}

    inline const Facet& operator*() { return *operator->(); }
    inline const Facet* operator->() { return static_cast<Facet*>(MeshEntityIterator::operator->()); }

  };

  /// A FacetFunction is a MeshFunction of topological codimension 1.

  template <class T> class FacetFunction : public MeshFunction<T>
  {
  public:

    FacetFunction(const Mesh& mesh) : MeshFunction<T>(mesh, mesh.topology().dim() - 1) {}

  };

}

#endif

// Copyright (C) 2009-2011 Andre Massing
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
// First added:  2009-09-11
// Last changed: 2011-08-23

#ifndef  meshprimitive_INC
#define  meshprimitive_INC

#ifdef HAS_CGAL

#include <CGAL/Bbox_3.h>

#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntityIterator.h>
#include <dolfin/mesh/SubsetIterator.h>
#include <dolfin/mesh/Vertex.h>

#include "PrimitiveTraits.h"

namespace dolfin
{

/// A template class which satisfies the Primitive concept required by CGALs
/// AABBTree template class

template <typename PrimitiveTrait>
class MeshPrimitive
{
public:

  typedef std::size_t Id;
  typedef typename PrimitiveTrait::Datum Datum;
  typedef typename PrimitiveTrait::K K;
  typedef typename PrimitiveTrait::Primitive Primitive;
  typedef typename K::Point_3 Point_3;

  /// Topological dimension of the MeshEntity
  static const std::size_t dim = PrimitiveTrait::dim;

  /// Static, so only reference to a mesh and entity index have to be saved
  static MeshEntity getEntity(const MeshPrimitive & p)
  { return MeshEntity(p.mesh(), dim, p.index); }

  const Mesh& mesh() const { return *_mesh; }

private:

  Id index; // this is what the AABB tree stores internally
  const Mesh * _mesh;

public:

  MeshPrimitive(): index(0), _mesh(0)   {} // default constructor needed

  /// Create a MeshPrimitive from a given MeshEntityIterator
  MeshPrimitive(MeshEntityIterator entity)
    : index(entity->index()), _mesh(&(entity->mesh())) {}

  /// Create a MeshPrimitive from a given SubsetIterator
  MeshPrimitive(SubsetIterator entity)
    : index(entity->index()), _mesh(&(entity->mesh())) {}

  Id id() const
  { return index;}

  // *Not* required by the CGAL primitive concept, but added for
  // efficieny easons. Explanation: We use a modified AABB_tree, in
  // which the local BBox functor class has been redefined to use the
  // bbox function of dolfin mesh entities.  Otherwise the bbox
  // function of the Datum object (see below) would have been used,
  // which means that we would have had to convert dolfin entities into
  // CGAL primitives only to initialize the tree, which is probably
  // very costly for 1 million of triangles.

  // CGAL::Bbox_3 bbox () const
  // { return MeshPrimitive<PrimitiveTrait>::getEntity(*this).bbox<K>(); }

  // Provides a reference point required by the Primitive concept of CGAL
  // AABB_tree. Uses conversion operator in dolfin::Point to create a certain
  // CGAL Point_3 type.
  Point_3 reference_point() const
  {
    return VertexIterator(MeshPrimitive<PrimitiveTrait>::getEntity(*this))->point();
  }

  Datum datum() const
  { return PrimitiveTraits<Primitive,K>::datum(MeshPrimitive<PrimitiveTrait>::getEntity(*this));}

};

}

#endif

#endif

// Copyright (C) 2019 Matthew Scroggs
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/types.h>
#include <dolfin/mesh/Mesh.h>
#include <ufc.h>

namespace dolfin
{
namespace fem
{

/// Element vector type identifier
enum class ElementVectorType : int
{
  scalar = 0,
  div = 1,
  curl = 2
};

/// Vertex arrangement type identifier
enum class VertexArrangementType : int
{
  none = -1,
  point = 0
};

/// Edge arrangement type identifier
enum class EdgeArrangementType : int
{
  none = -1,
  interval = 0
};

/// Face arrangement type identifier
enum class FaceArrangementType : int
{
  none = -1,
  triangle = 0,
  quadrilateral = 1
};

/// Volume arrangement type identifier
enum class VolumeArrangementType : int
{
  none = -1,
  tetrahedron = 0,
  hexahedron = 1
};

/// Class that contains information about the dof arrangement on entities
/// TODO: Move all this into ElementDofLayout
class EntityArrangementTypes
{
public:
  /// Construction
  EntityArrangementTypes(const ufc_dofmap& dofmap,
                         const mesh::CellType cell_type);

  /// Get the arrangement type on vertices
  /// @return The vertex arrangement type
  VertexArrangementType vertex_type() const { return _vertex_type; }

  /// Get the arrangement type on edges
  /// @return The edge arrangement type
  EdgeArrangementType edge_type() const { return _edge_type; }

  /// Get the arrangement type on faces
  /// @return The face arrangement type
  FaceArrangementType face_type() const { return _face_type; }

  /// Get the arrangement type on volumes
  /// @return The volume arrangement type
  VolumeArrangementType volume_type() const { return _volume_type; }

  /// Get the vector type of the element
  /// @return The vector type of the element
  ElementVectorType element_type() const { return _element_type; }

  /// Get the blocksize on an entity of a dimension
  /// @param[in] The dimension of the entity
  /// @return The block size
  int get_block_size(const int dim) const;

private:
  fem::VertexArrangementType _vertex_type;
  fem::EdgeArrangementType _edge_type;
  fem::FaceArrangementType _face_type;
  fem::VolumeArrangementType _volume_type;
  fem::ElementVectorType _element_type;
};

} // namespace fem
} // namespace dolfin

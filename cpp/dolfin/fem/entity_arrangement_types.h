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

class EntityArrangementTypes
{
public:
  EntityArrangementTypes(const ufc_dofmap& dofmap,
                         const mesh::CellType cell_type);
  VertexArrangementType vertex_type() const { return _vertex_type; }
  EdgeArrangementType edge_type() const { return _edge_type; }
  FaceArrangementType face_type() const { return _face_type; }
  VolumeArrangementType volume_type() const { return _volume_type; }
  ElementVectorType element_type() const { return _element_type; }
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

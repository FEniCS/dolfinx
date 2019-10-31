// Copyright (C) 2006-2019 Matthew Scroggs
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "entity_arrangement_types.h"
#include <dolfin/common/log.h>

namespace
{
//-----------------------------------------------------------------------------
int _vertex_arrangement_blocksize(dolfin::fem::ElementVectorType type,
                                  const int element_dim)
{
  return 1;
}
//-----------------------------------------------------------------------------
int _edge_arrangement_blocksize(dolfin::fem::ElementVectorType type,
                                const int element_dim)
{
  return 1;
}
//-----------------------------------------------------------------------------
int _face_arrangement_blocksize(dolfin::fem::ElementVectorType type,
                                const int element_dim)
{
  switch (type)
  {
  case dolfin::fem::ElementVectorType::scalar:
    return 1;
  case dolfin::fem::ElementVectorType::curl:
    return 2;
  case dolfin::fem::ElementVectorType::div:
    if (element_dim == 3)
      return 1;
    return 2;
  case dolfin::fem::ElementVectorType::ein:
    throw std::runtime_error("HEin not yet implemented.");
  case dolfin::fem::ElementVectorType::divdiv:
    throw std::runtime_error("HDivDiv not yet implemented.");
  }
  // Should not reach this point
  return 0;
}
//-----------------------------------------------------------------------------
int _volume_arrangement_blocksize(dolfin::fem::ElementVectorType type,
                                  const int element_dim)
{
  switch (type)
  {
  case dolfin::fem::ElementVectorType::scalar:
    return 1;
  case dolfin::fem::ElementVectorType::curl:
    return 3;
  case dolfin::fem::ElementVectorType::div:
    return 3;
  case dolfin::fem::ElementVectorType::ein:
    throw std::runtime_error("HEin not yet implemented.");
  case dolfin::fem::ElementVectorType::divdiv:
    throw std::runtime_error("HDivDiv not yet implemented.");
  }
  // Should not reach this point
  return 0;
}
//-----------------------------------------------------------------------------
} // namespace

namespace dolfin
{
namespace fem
{
//-----------------------------------------------------------------------------
EntityArrangementTypes::EntityArrangementTypes(const ufc_dofmap& dofmap,
                                               const mesh::CellType cell_type)
{
  _vertex_type = VertexArrangementType::none;
  _edge_type = EdgeArrangementType::none;
  _face_type = FaceArrangementType::none;
  _volume_type = VolumeArrangementType::none;
  switch (cell_type)
  {
  case (mesh::CellType::point):
    _vertex_type = VertexArrangementType::point;
    break;
  case (mesh::CellType::interval):
    _vertex_type = VertexArrangementType::point;
    _edge_type = EdgeArrangementType::interval;
    break;
  case (mesh::CellType::triangle):
    _vertex_type = VertexArrangementType::point;
    _edge_type = EdgeArrangementType::interval;
    _face_type = FaceArrangementType::triangle;
    break;
  case (mesh::CellType::quadrilateral):
    _vertex_type = VertexArrangementType::point;
    _edge_type = EdgeArrangementType::interval;
    _face_type = FaceArrangementType::quadrilateral;
    break;
  case (mesh::CellType::tetrahedron):
    _vertex_type = VertexArrangementType::point;
    _edge_type = EdgeArrangementType::interval;
    _face_type = FaceArrangementType::triangle;
    _volume_type = VolumeArrangementType::tetrahedron;
    break;
  case (mesh::CellType::hexahedron):
    _vertex_type = VertexArrangementType::point;
    _edge_type = EdgeArrangementType::interval;
    _face_type = FaceArrangementType::quadrilateral;
    _volume_type = VolumeArrangementType::hexahedron;
    break;
  default:
    throw std::runtime_error("Unrecognised cell type.");
  }

  if (dofmap.face_arrangement_type == 3)
    _face_type = FaceArrangementType::triangle;
  else if (dofmap.face_arrangement_type == 4)
    _face_type = FaceArrangementType::quadrilateral;

  if (dofmap.volume_arrangement_type == 3)
    _volume_type = VolumeArrangementType::tetrahedron;
  else if (dofmap.volume_arrangement_type == 4)
    _volume_type = VolumeArrangementType::hexahedron;

  /* From ufc:
      mixed = -1, L2 = 0, H1 = 1, H2 = 2,
      HCurl = 3, HDiv = 4, HEin = 5, HDivDiv = 6,
  */
  if (dofmap.sobolev_space_type == 0)
    _element_type = ElementVectorType::scalar;
  else if (dofmap.sobolev_space_type == 1)
    _element_type = ElementVectorType::scalar;
  else if (dofmap.sobolev_space_type == 2)
    _element_type = ElementVectorType::scalar;
  else if (dofmap.sobolev_space_type == 3)
    _element_type = ElementVectorType::curl;
  else if (dofmap.sobolev_space_type == 4)
    _element_type = ElementVectorType::div;
  else if (dofmap.sobolev_space_type == 5)
    _element_type = ElementVectorType::ein; // TODO: what is this
  else if (dofmap.sobolev_space_type == 6)
    _element_type = ElementVectorType::divdiv; // TODO: what is this
}
//-----------------------------------------------------------------------------
int EntityArrangementTypes::get_block_size(const int dim,
                                           const int element_dim) const
{
  if (dim == 0)
  {
    return _vertex_arrangement_blocksize(_element_type, element_dim);
  }
  if (dim == 1)
  {
    return _edge_arrangement_blocksize(_element_type, element_dim);
  }
  if (dim == 2)
  {
    return _face_arrangement_blocksize(_element_type, element_dim);
  }
  if (dim == 3)
  {
    return _volume_arrangement_blocksize(_element_type, element_dim);
  }
  throw std::runtime_error("Unrecognised arrangement type.");
}
//-----------------------------------------------------------------------------
} // namespace fem
} // namespace dolfin

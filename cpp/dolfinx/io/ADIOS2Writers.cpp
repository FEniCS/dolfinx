// Copyright (C) 2021 Jørgen S. Dokken and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include "ADIOS2Writers.h"
#include "cells.h"
#include <pugixml.hpp>
#include <string>
#include <vector>

using namespace dolfinx;
using namespace dolfinx::io;

//-----------------------------------------------------------------------------
std::string impl_fides::to_fides_cell(mesh::CellType type)
{
  switch (type)
  {
  case mesh::CellType::point:
    return "vertex";
  case mesh::CellType::interval:
    return "line";
  case mesh::CellType::triangle:
    return "triangle";
  case mesh::CellType::tetrahedron:
    return "tetrahedron";
  case mesh::CellType::quadrilateral:
    return "quad";
  case mesh::CellType::pyramid:
    return "pyramid";
  case mesh::CellType::prism:
    return "wedge";
  case mesh::CellType::hexahedron:
    return "hexahedron";
  default:
    throw std::runtime_error("Unknown cell type.");
  }
}
//-----------------------------------------------------------------------------
std::stringstream
io::impl_vtx::create_vtk_schema(const std::vector<std::string>& point_data,
                                const std::vector<std::string>& cell_data)
{
  // Create XML
  pugi::xml_document xml_schema;
  pugi::xml_node vtk_node = xml_schema.append_child("VTKFile");
  vtk_node.append_attribute("type") = "UnstructuredGrid";
  vtk_node.append_attribute("version") = "0.1";
  pugi::xml_node unstructured = vtk_node.append_child("UnstructuredGrid");

  // -- Piece

  pugi::xml_node piece = unstructured.append_child("Piece");

  // Add mesh attributes
  piece.append_attribute("NumberOfPoints") = "NumberOfNodes";
  piece.append_attribute("NumberOfCells") = "NumberOfCells";

  // -- Points

  // Add point information
  pugi::xml_node xml_geometry = piece.append_child("Points");
  pugi::xml_node xml_vertices = xml_geometry.append_child("DataArray");
  xml_vertices.append_attribute("Name") = "geometry";

  // -- Cells

  pugi::xml_node xml_topology = piece.append_child("Cells");
  xml_topology.append_child("DataArray").append_attribute("Name")
      = "connectivity";
  xml_topology.append_child("DataArray").append_attribute("Name") = "types";

  // -- PointData

  pugi::xml_node xml_pointdata = piece.append_child("PointData");

  // Stepping info for time dependency
  pugi::xml_node item_time = xml_pointdata.append_child("DataArray");
  item_time.append_attribute("Name") = "TIME";
  item_time.append_child(pugi::node_pcdata).set_value("step");

  pugi::xml_node item_idx = xml_pointdata.append_child("DataArray");
  item_idx.append_attribute("Name") = "vtkOriginalPointIds";
  pugi::xml_node item_ghost = xml_pointdata.append_child("DataArray");
  item_ghost.append_attribute("Name") = "vtkGhostType";
  for (auto& name : point_data)
  {
    pugi::xml_node item = xml_pointdata.append_child("DataArray");
    item.append_attribute("Name") = name.c_str();
  }

  // -- CellData

  if (!cell_data.empty())
  {
    pugi::xml_node xml_celldata = piece.append_child("CellData");
    for (auto& name : cell_data)
    {
      pugi::xml_node item = xml_celldata.append_child("DataArray");
      item.append_attribute("Name") = name.c_str();
    }
  }

  std::stringstream ss;
  xml_schema.save(ss, "  ");
  return ss;
}
//-----------------------------------------------------------------------------

#endif

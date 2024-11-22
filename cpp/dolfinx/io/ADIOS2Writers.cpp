// Copyright (C) 2021-2023 Jørgen S. Dokken and Garth N. Wells
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
ADIOS2Writer::ADIOS2Writer(MPI_Comm comm, const std::filesystem::path& filename,
                           std::string tag, std::string engine)
    : _adios(std::make_unique<adios2::ADIOS>(comm)),
      _io(std::make_unique<adios2::IO>(_adios->DeclareIO(tag)))
{
  _io->SetEngine(engine);
  _engine = std::make_unique<adios2::Engine>(
      _io->Open(filename, adios2::Mode::Write));
}
//-----------------------------------------------------------------------------
ADIOS2Writer::~ADIOS2Writer() { close(); }
//-----------------------------------------------------------------------------
void ADIOS2Writer::close()
{
  assert(_engine);
  // The reason this looks odd is that ADIOS2 uses `operator bool()`
  // to test if the engine is open
  if (*_engine)
    _engine->Close();
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

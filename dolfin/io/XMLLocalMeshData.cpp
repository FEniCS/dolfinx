// Copyright (C) 2002-2011 Anders Logg, Ola Skavhaug and Garth N. Wells
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
// First added:  2002-12-06
// Last changed: 2006-10-16

#include <map>
#include <iomanip>
#include <iostream>
#include <vector>
#include <boost/assign/list_of.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include "pugixml.hpp"

#include "dolfin/common/Array.h"
#include "dolfin/common/MPI.h"
#include "dolfin/la/GenericVector.h"
#include "dolfin/mesh/Cell.h"
#include "dolfin/mesh/CellType.h"
#include "dolfin/mesh/LocalMeshData.h"
#include "dolfin/mesh/Mesh.h"
#include "dolfin/mesh/MeshData.h"
#include "dolfin/mesh/MeshEditor.h"
#include "dolfin/mesh/Point.h"
#include "dolfin/mesh/Vertex.h"
#include "XMLIndent.h"
#include "XMLMeshFunction.h"
#include "XMLLocalMeshData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void XMLLocalMeshData::read(LocalMeshData& mesh_data, const pugi::xml_node xml_dolfin)
{
  std::cout << "Using new-style local mesh data reading." << std::endl;

  // Clear mesh data
  mesh_data.clear();

  // Check that we have an XML mesh
  pugi::xml_node xml_mesh(0);
  if (xml_dolfin)
  {
    xml_mesh = xml_dolfin.child("mesh");
    if (!xml_mesh)
      error("Not a DOLFIN Mesh file.");
  }

  // Some process rank checks
  if (!xml_dolfin && MPI::process_number() == 0)
    error("XMLLocalMeshData::read must read from root process.");
  if (xml_dolfin && MPI::process_number() != 0)
    error("XMLLocalMeshData::read must read from root process only.");

  // Get geometric and topological dimensions and broadcast from root
  if (xml_mesh)
  {
    // Get cell type and geometric dimension
    const std::string cell_type_str = xml_mesh.attribute("celltype").value();
    mesh_data.gdim = xml_mesh.attribute("dim").as_uint();

    // Get topological dimension and number of vertices per cell
    boost::scoped_ptr<CellType> cell_type(CellType::create(cell_type_str));
    mesh_data.tdim = cell_type->dim();
    mesh_data.num_vertices_per_cell = cell_type->num_entities(0);

    // Read number of global vertices
    pugi::xml_node xml_vertices = xml_mesh.child("vertices");
    assert(xml_vertices);
    mesh_data.num_global_vertices = xml_vertices.attribute("size").as_uint();

    // Read number of global cells
    pugi::xml_node xml_cells = xml_mesh.child("cells");
    assert(xml_vertices);
    mesh_data.num_global_cells = xml_cells.attribute("size").as_uint();
  }
  MPI::broadcast(mesh_data.gdim, 0);
  MPI::broadcast(mesh_data.tdim, 0);
  MPI::broadcast(mesh_data.num_global_vertices, 0);
  MPI::broadcast(mesh_data.num_global_cells, 0);
  MPI::broadcast(mesh_data.num_vertices_per_cell, 0);

  cout << "Dims: " << mesh_data.gdim << "  " << mesh_data.tdim <<  endl;

  // Read vertex data
  if (xml_mesh)
  {
    // Get vertices xml node
    pugi::xml_node xml_vertices = xml_mesh.child("vertices");
    assert(xml_vertices);

    // Read vertex data
    const unsigned int num_vertices = xml_vertices.attribute("size").as_uint();

    // Iterate over vertices and add to mesh
    mesh_data.vertex_indices.reserve(num_vertices);
    mesh_data.vertex_coordinates.reserve(num_vertices);
    for (pugi::xml_node_iterator it = xml_vertices.begin(); it != xml_vertices.end(); ++it)
    {
      const unsigned int index = it->attribute("index").as_uint();
      std::vector<double> coordinate;
      switch (mesh_data.gdim)
      {
      case 1:
        coordinate = boost::assign::list_of(it->attribute("x").as_double());
        break;
      case 2:
        coordinate = boost::assign::list_of(it->attribute("x").as_double())
                                           (it->attribute("y").as_double());
        break;
      case 3:
        coordinate = boost::assign::list_of(it->attribute("x").as_double())
                                           (it->attribute("y").as_double())
                                           (it->attribute("z").as_double());
      break;
      default:
        error("Geometric dimension of mesh must be 1, 2 or 3.");
      }
      mesh_data.vertex_coordinates.push_back(coordinate);
      mesh_data.vertex_indices.push_back(index);
    }
  }
  else
  {
    mesh_data.vertex_indices.clear();
    mesh_data.vertex_coordinates.clear();
  }
  cout << "Finished vertex input " << mesh_data.tdim <<  endl;

  // Read cells data
  if (xml_mesh)
  {
    // Get cells node
    pugi::xml_node xml_cells = xml_mesh.child("cells");
    assert(xml_cells);

    // Get number of cells and init editor
    const unsigned int num_cells = xml_cells.attribute("size").as_uint();

    // Get cell type and geometric dimension
    const std::string cell_type_str = xml_mesh.attribute("celltype").value();
    boost::scoped_ptr<const CellType> cell_type(CellType::create(cell_type_str));
    const unsigned int num_vertices_per_cell = cell_type->num_vertices(0);

    mesh_data.cell_vertices.reserve(num_cells);
    mesh_data.global_cell_indices.reserve(num_cells);

    // Create list of vertex index attribute names
    std::vector<std::string> v_str(num_vertices_per_cell);
    for (uint i = 0; i < num_vertices_per_cell; ++i)
      v_str[i] = "v" + boost::lexical_cast<std::string, unsigned int>(i);

    // Iterate over cells and add to mesh
    std::vector<unsigned int> v(num_vertices_per_cell);
    for (pugi::xml_node_iterator it = xml_cells.begin(); it != xml_cells.end(); ++it)
    {
      const unsigned int index = it->attribute("index").as_uint();
      for (unsigned int i = 0; i < num_vertices_per_cell; ++i)
        v[i] = it->attribute(v_str[i].c_str()).as_uint();

      mesh_data.cell_vertices.push_back(v);
      mesh_data.global_cell_indices.push_back(index);
    }
  }
  else
  {
    mesh_data.cell_vertices.clear();
    mesh_data.global_cell_indices.clear();
  }
  cout << "Finished cell input " << mesh_data.tdim <<  endl;
}
//-----------------------------------------------------------------------------
void XMLLocalMeshData::write(const LocalMeshData& mesh_data, std::ostream& outfile,
                    uint indentation_level)
{
  error("Writing of mesh LocalData to XML files is not supported.");
}
//-----------------------------------------------------------------------------

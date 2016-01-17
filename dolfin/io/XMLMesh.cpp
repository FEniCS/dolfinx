// Copyright (C) 2011-2013 Garth N. Wells
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
// Modified by Anders Logg 2011
//
// First added:  2002-12-06
// Last changed: 2014-02-06

#include <map>
#include <memory>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <boost/format.hpp>

#include "pugixml.hpp"

#include "dolfin/common/MPI.h"
#include "dolfin/common/NoDeleter.h"
#include "dolfin/geometry/Point.h"
#include "dolfin/la/GenericVector.h"
#include "dolfin/mesh/Cell.h"
#include "dolfin/mesh/CellType.h"
#include "dolfin/mesh/LocalMeshData.h"
#include "dolfin/mesh/Mesh.h"
#include "dolfin/mesh/MeshData.h"
#include "dolfin/mesh/MeshEditor.h"
#include "dolfin/mesh/Vertex.h"
#include "dolfin/mesh/MeshFunction.h"
#include "XMLMeshFunction.h"
#include "XMLMeshValueCollection.h"
#include "XMLMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void XMLMesh::read(Mesh& mesh, const pugi::xml_node xml_dolfin)
{
  // Get mesh node
  const pugi::xml_node mesh_node = xml_dolfin.child("mesh");
  if (!mesh_node)
  {
    dolfin_error("XMLMesh.cpp",
                 "read mesh from XML file",
                 "Not a DOLFIN XML Mesh file");
  }

  // Read mesh
  read_mesh(mesh, mesh_node);

  // Read mesh data (if any)
  read_data(mesh.data(), mesh, mesh_node);

  // Read mesh domains (if any)
  read_domains(mesh.domains(), mesh, mesh_node);
}
//-----------------------------------------------------------------------------
void XMLMesh::write(const Mesh& mesh, pugi::xml_node xml_node)
{
  // Add mesh node
  pugi::xml_node mesh_node = xml_node.append_child("mesh");

  // Write mesh
  write_mesh(mesh, mesh_node);

  // Write mesh data (if any)
  write_data(mesh, mesh.data(), mesh_node);

  // Write mesh markers (if any)
  write_domains(mesh, mesh.domains(), mesh_node);
}
//-----------------------------------------------------------------------------
void XMLMesh::read_mesh(Mesh& mesh, const pugi::xml_node mesh_node)
{
  // Get cell type and geometric dimension
  const std::string cell_type_str = mesh_node.attribute("celltype").value();
  const std::size_t gdim = mesh_node.attribute("dim").as_uint();

  // Get topological dimension
  std::unique_ptr<CellType> cell_type(CellType::create(cell_type_str));
  const std::size_t tdim = cell_type->dim();

  // Create mesh for editing
  MeshEditor editor;
  editor.open(mesh, cell_type_str, tdim, gdim);

  // Get vertices xml node
  pugi::xml_node xml_vertices = mesh_node.child("vertices");
  dolfin_assert(xml_vertices);

  // Get number of vertices and init editor
  const std::size_t num_vertices = xml_vertices.attribute("size").as_uint();
  editor.init_vertices_global(num_vertices, num_vertices);

  // Iterate over vertices and add to mesh
  Point p;
  for (pugi::xml_node_iterator it = xml_vertices.begin();
       it != xml_vertices.end(); ++it)
  {
    const std::size_t index = it->attribute("index").as_uint();
    p[0] = it->attribute("x").as_double();
    p[1] = it->attribute("y").as_double();
    p[2] = it->attribute("z").as_double();
    editor.add_vertex(index, p);
  }

  // Get cells node
  pugi::xml_node xml_cells = mesh_node.child("cells");
  dolfin_assert(xml_cells);

  // Get number of cells and init editor
  const std::size_t num_cells = xml_cells.attribute("size").as_uint();
  editor.init_cells_global(num_cells, num_cells);

  // Create list of vertex index attribute names
  const unsigned int num_vertices_per_cell = cell_type->num_vertices(tdim);
  std::vector<std::string> v_str(num_vertices_per_cell);
  for (std::size_t i = 0; i < num_vertices_per_cell; ++i)
    v_str[i] = "v" + std::to_string(i);

  // Iterate over cells and add to mesh
  std::vector<std::size_t> v(num_vertices_per_cell);
  for (pugi::xml_node_iterator it = xml_cells.begin(); it != xml_cells.end();
       ++it)
  {
    const std::size_t index = it->attribute("index").as_uint();
    for (unsigned int i = 0; i < num_vertices_per_cell; ++i)
      v[i] = it->attribute(v_str[i].c_str()).as_uint();
    editor.add_cell(index, v);
  }

  // Close mesh editor
  editor.close();
}
//-----------------------------------------------------------------------------
void XMLMesh::read_data(MeshData& data, const Mesh& mesh,
                        const pugi::xml_node mesh_node)
{
  // Check if we have any mesh data
  const pugi::xml_node xml_data = mesh_node.child("data");
  if (!xml_data)
    return;

  // Iterate over data
  for (pugi::xml_node_iterator it = xml_data.begin();
       it != xml_data.end(); ++it)
  {
    // Get node name
    const std::string node_name = it->name();

    // Read data stored as a MeshFunction (new style)
    if (node_name == "mesh_function")
    {
      // Create MeshFunction to read data into
      auto _mesh = reference_to_no_delete_pointer(mesh);
      MeshFunction<std::size_t> mf(_mesh);

      // Read  MeshFunction
      //const std::string data_type = it->attribute("type").value();
      XMLMeshFunction::read(mf, "uint", *it);

      // Create mesh domain array
      std::vector<std::size_t>& _data
        = data.create_array(mf.name(), mf.dim());
      _data.resize(mf.size());

      // Copy MeshFunction into MeshDomain array
      for (std::size_t i = 0; i < _data.size(); ++i)
        _data[i] = mf[i];
    }
    else if (node_name != "data_entry")
    {
      dolfin_error("XMLMesh.cpp",
                   "read mesh data from XML file",
                   "Expecting XML node <data_entry> but got <%s>",
                   node_name.c_str());
    }
    else // Old-style storage
    {
      // Get name of data set
      const std::string data_set_name = it->attribute("name").value();

      // Check that there is only one data set
      if (it->first_child().next_sibling())
      {
        dolfin_error("XMLMesh.cpp",
                     "read mesh data from XML file",
                     "XML file contains too many data sets");
      }

      // Get type of data set
      pugi::xml_node data_set = it->first_child();
      const std::string data_set_type = data_set.name();
      if (data_set_type == "array")
      {
        dolfin_error("XMLMesh.cpp",
                     "read mesh data from XML file",
                     "Only MeshFunction data can be read");
      }
      else if (data_set_type == "mesh_function")
      {
        // Create MeshFunction to read data into
        auto _mesh = reference_to_no_delete_pointer(mesh);
        MeshFunction<std::size_t> mf(_mesh);

        // Read  MeshFunction
        const std::string data_type = data_set.attribute("type").value();
        XMLMeshFunction::read(mf, data_type, *it);

        // Create mesh domain array
        std::vector<std::size_t>& _data
          = data.create_array(data_set_name, mf.dim());
        _data.resize(mf.size());

        // Copy MeshFunction into MeshDomain array
        for (std::size_t i = 0; i < _data.size(); ++i)
          _data[i] = mf[i];
      }
      else if (data_set_type == "meshfunction")
      {
        dolfin_error("XMLMesh.cpp",
                     "read mesh data from XML file",
                     "The XML tag <meshfunction> has been changed to <mesh_function>");
      }
      else
      {
        dolfin_error("XMLMesh.cpp",
                     "read mesh data from XML file",
                     "Reading of MeshData \"%s\" is not yet supported",
                     data_set_type.c_str());
      }
    }
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::read_domains(MeshDomains& domains, const Mesh& mesh,
                           const pugi::xml_node mesh_node)
{
  // Check if we have any domains
  const pugi::xml_node xml_domains = mesh_node.child("domains");
  if (!xml_domains)
    return;

  // Iterate over data
  for (pugi::xml_node_iterator it = xml_domains.begin();
       it != xml_domains.end(); ++it)
  {
    // Check that node is <mesh_value_collection>
    const std::string node_name = it->name();
    if (node_name != "mesh_value_collection")
    {
      dolfin_error("XMLMesh.cpp",
                   "read mesh domains from XML file",
                   "Expecting XML node <mesh_value_collection> but got <%s>",
                   node_name.c_str());
    }

    // Get attributes
    const std::string type = it->attribute("type").value();
    const std::size_t dim = it->attribute("dim").as_uint();

    // Check that the type is uint
    if (type != "uint")
    {
      dolfin_error("XMLMesh.cpp",
                   "read mesh domains from XML file",
                   "Mesh domains must be marked as uint, not %s",
                   type.c_str());
    }

    // Initialise mesh entities
    mesh.init(dim);

    // Read data into a mesh value collection
    std::shared_ptr<const Mesh> _mesh = reference_to_no_delete_pointer(mesh);
    MeshValueCollection<std::size_t> mvc(_mesh);
    XMLMeshValueCollection::read(mvc, type, *it);

    // Get mesh value collection data
    const std::map<std::pair<std::size_t, std::size_t>, std::size_t>&
      values = mvc.values();

    // Get mesh domain data and fill
    std::map<std::size_t, std::size_t>& markers
      = domains.markers(dim);
    std::map<std::pair<std::size_t, std::size_t>,
             std::size_t>::const_iterator entry;
    if (dim != mesh.topology().dim())
    {
      for (entry = values.begin(); entry != values.end(); ++entry)
      {
        const Cell cell(mesh, entry->first.first);
        const std::size_t entity_index
          = cell.entities(dim)[entry->first.second];
        markers[entity_index] = entry->second;
      }
    }
    else
    {
      // Special case for cells
      for (entry = values.begin(); entry != values.end(); ++entry)
        markers[entry->first.first] = entry->second;
    }
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::read_domain_data(LocalMeshData& mesh_data,
                               const pugi::xml_node xml_dolfin)
{
  // Get mesh node
  const pugi::xml_node mesh_node = xml_dolfin.child("mesh");
  if (!mesh_node)
  {
    dolfin_error("XMLMesh.cpp",
                 "read mesh from XML file",
                 "Not a DOLFIN XML Mesh file");
  }


  // Check if we have any domains
  const pugi::xml_node xml_domains = mesh_node.child("domains");
  if (!xml_domains)
  {
    mesh_data.domain_data.clear();
    return;
  }

  // Iterate over data
  for (pugi::xml_node_iterator xml_domain = xml_domains.begin();
       xml_domain != xml_domains.end(); ++xml_domain)
  {
    // Check that node is <mesh_value_collection>
    const std::string node_name = xml_domain->name();
    if (node_name != "mesh_value_collection")
    {
      dolfin_error("XMLMesh.cpp",
                   "read mesh domains from XML file",
                   "Expecting XML node <mesh_value_collection> but got <%s>",
                   node_name.c_str());
    }

    // Get attributes
    const std::string type = xml_domain->attribute("type").value();
    const std::size_t dim = xml_domain->attribute("dim").as_uint();

    // Check that the type is uint
    if (type != "uint")
    {
      dolfin_error("XMLMesh.cpp",
                   "read mesh domains from XML file",
                   "Mesh domains must be marked as uint, not %s",
                   type.c_str());
    }

    // Get domain data
    std::vector<std::pair<std::pair<std::size_t, std::size_t>, std::size_t>>&
      domain_data = mesh_data.domain_data[dim] ;

    pugi::xml_node_iterator xml_domain_dim;
    for (xml_domain_dim = xml_domain->begin();
         xml_domain_dim != xml_domain->end(); ++xml_domain_dim)
    {
      const std::size_t cell_index
        = xml_domain_dim->attribute("cell_index").as_uint();
      const std::size_t local_entity
          = xml_domain_dim->attribute("local_entity").as_uint();
      const std::size_t value = xml_domain_dim->attribute("value").as_uint();

      domain_data.push_back({{cell_index, local_entity}, value});
    }
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::read_array_uint(std::vector<std::size_t>& array,
                              const pugi::xml_node xml_array)
{
  // Check that we have an array
  const std::string name = xml_array.name();
  if (name != "array")
  {
    dolfin_error("XMLMesh.cpp",
                 "read mesh array data from XML file",
                 "Expecting an XML array node");
  }

  // Check type is unit
  const std::string type = xml_array.attribute("type").value();
  if (type != "uint")
  {
    dolfin_error("XMLMesh.cpp",
                 "read mesh array data from XML file",
                 "Expecting an XML array node");
  }

  // Get size and resize vector
  const std::size_t size = xml_array.attribute("size").as_uint();
  array.resize(size);

  // Iterate over array entries
  for (pugi::xml_node_iterator it = xml_array.begin(); it !=xml_array.end();
       ++it)
  {
    const std::size_t index = it->attribute("index").as_uint();
    const double value = it->attribute("value").as_uint();
    dolfin_assert(index < size);
    array[index] = value;
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::write_mesh(const Mesh& mesh, pugi::xml_node mesh_node)
{
  // Add mesh attributes
  const CellType::Type _cell_type = mesh.type().cell_type();
  const std::string cell_type = CellType::type2string(_cell_type);
  mesh_node.append_attribute("celltype") = cell_type.c_str();
  mesh_node.append_attribute("dim") = (unsigned int) mesh.geometry().dim();

  // Add vertices node
  pugi::xml_node vertices_node = mesh_node.append_child("vertices");
  vertices_node.append_attribute("size") = (unsigned int) mesh.num_vertices();

  // Write each vertex
  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    pugi::xml_node vertex_node = vertices_node.append_child("vertex");
    vertex_node.append_attribute("index") = (unsigned int) v->index();

    const Point p = v->point();
    switch (mesh.geometry().dim())
    {
      case 1:
        vertex_node.append_attribute("x")
            = boost::str(boost::format("%.15e") % p.x()).c_str();
        break;
      case 2:
        vertex_node.append_attribute("x")
            = boost::str(boost::format("%.15e") % p.x()).c_str();
        vertex_node.append_attribute("y")
            = boost::str(boost::format("%.15e") % p.y()).c_str();
        break;
      case 3:
        vertex_node.append_attribute("x")
            = boost::str(boost::format("%.15e") % p.x()).c_str();
        vertex_node.append_attribute("y")
            = boost::str(boost::format("%.15e") % p.y()).c_str();
        vertex_node.append_attribute("z")
            = boost::str(boost::format("%.15e") % p.z()).c_str();
        break;
      default:
        dolfin_error("XMLMesh.cpp",
                     "write mesh to XML file",
                     "The XML mesh file format only supports 1D, 2D and 3D meshes");
    }
  }

  // Add cells node
  pugi::xml_node cells_node = mesh_node.append_child("cells");
  cells_node.append_attribute("size") = (unsigned int) mesh.num_cells();

  // Add each cell
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    pugi::xml_node cell_node = cells_node.append_child(cell_type.c_str());
    cell_node.append_attribute("index") = (unsigned int) c->index();

    const unsigned int* vertices = c->entities(0);
    dolfin_assert(vertices);

    switch (_cell_type)
    {
    case CellType::interval:
      cell_node.append_attribute("v0") = (unsigned int) vertices[0];
      cell_node.append_attribute("v1") = (unsigned int) vertices[1];
      break;
    case CellType::triangle:
      cell_node.append_attribute("v0") = (unsigned int) vertices[0];
      cell_node.append_attribute("v1") = (unsigned int) vertices[1];
      cell_node.append_attribute("v2") = (unsigned int) vertices[2];
      break;
    case CellType::tetrahedron:
      cell_node.append_attribute("v0") = (unsigned int) vertices[0];
      cell_node.append_attribute("v1") = (unsigned int) vertices[1];
      cell_node.append_attribute("v2") = (unsigned int) vertices[2];
      cell_node.append_attribute("v3") = (unsigned int) vertices[3];
      break;
    default:
      dolfin_error("XMLMesh.cpp",
                   "write mesh to XML file",
                   "Unknown cell type (%u)", _cell_type);
    }
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::write_data(const Mesh& mesh, const MeshData& data,
                         pugi::xml_node mesh_node)
{
  // Check if there is any data to write
  if (data._arrays.empty())
    return;

  // Add mesh data node
  pugi::xml_node mesh_data_node = mesh_node.append_child("data");

  // Write arrays
  typedef std::vector<std::map<std::string,
                               std::vector<std::size_t>>>::const_iterator array_iterator_d;
  typedef std::map<std::string,
                   std::vector<std::size_t>>::const_iterator array_iterator;
  for (array_iterator_d it_d = data._arrays.begin();
       it_d != data._arrays.end(); ++it_d)
  {
    const std::size_t dim =  it_d - data._arrays.begin();
    for (array_iterator it = it_d->begin(); it != it_d->end(); ++it)
    {
      // Data set name and array
      std::string name = it->first;
      const std::vector<std::size_t>& array = it->second;

      // Check data length
      if (array.size() != mesh.num_entities(dim))
      {
        dolfin_error("XMLMesh.cpp",
                     "write mesh data to XML file",
                     "Data array length does not match number of mesh entities");
      }

      // Copy data into MeshFunction
      auto _mesh = reference_to_no_delete_pointer(mesh);
      MeshFunction<std::size_t> mf(_mesh, dim);
      mf.rename(name, name);
      for (std::size_t i = 0; i < mf.size(); ++i)
        mf[i] = array[i];

      // Write MeshFunction
      XMLMeshFunction::write(mf, "uint", mesh_data_node, false);
    }
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::write_domains(const Mesh& mesh, const MeshDomains& domains,
                            pugi::xml_node mesh_node)
{
  // Check if there is any data to write
  if (domains.is_empty())
    return;

  // Add mesh domains node
  pugi::xml_node domains_node = mesh_node.append_child("domains");

  // Write mesh markers
  for (std::size_t d = 0; d <= domains.max_dim(); d++)
  {
    if (!domains.markers(d).empty())
    {
      const std::map<std::size_t, std::size_t>& domain = domains.markers(d);

      auto _mesh = reference_to_no_delete_pointer(mesh);
      MeshValueCollection<std::size_t> collection(_mesh, d);
      std::map<std::size_t, std::size_t>::const_iterator it;
      for (it = domain.begin(); it != domain.end(); ++it)
        collection.set_value(it->first, it->second);
      XMLMeshValueCollection::write(collection, "uint", domains_node);
    }
  }
}
//-----------------------------------------------------------------------------

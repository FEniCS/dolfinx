// Copyright (C) 2009 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-06
// Last changed:  2009-03-16

#include <cstring>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshData.h>
#include "XMLArray.h"
#include "XMLMeshData.h"
#include "NewXMLMesh.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NewXMLMesh::NewXMLMesh(Mesh& mesh, NewXMLFile& parser) 
  : XMLHandler(parser), _mesh(mesh), state(OUTSIDE), f(0), a(0),
    mesh_coord(0), uint_array(0), xml_array(0), xml_mesh_data(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewXMLMesh::~NewXMLMesh()
{
  delete mesh_coord;
  delete uint_array;
  delete xml_mesh_data;
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewXMLMesh::start_element(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state )
  {
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "mesh") == 0 )
    {
      read_mesh_tag(name, attrs);
    }
    
    break;

  case INSIDE_MESH:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "vertices") == 0 )
    {
      read_vertices(name, attrs);
      state = INSIDE_VERTICES;
    }
    else if ( xmlStrcasecmp(name, (xmlChar *) "cells") == 0 )
    {
      read_cells(name, attrs);
      state = INSIDE_CELLS;
    }
    else if ( xmlStrcasecmp(name, (xmlChar *) "data") == 0 )
    {
      read_mesh_data(name, attrs);
      state = INSIDE_MESH;
    }
    else if ( xmlStrcasecmp(name, (xmlChar *) "coordinates") == 0 )
    {
      state = INSIDE_COORDINATES;
    }

    break;
    
  case INSIDE_VERTICES:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "vertex") == 0 )
      read_vertex(name, attrs);

    break;
    
  case INSIDE_CELLS:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "interval") == 0 )
      read_interval(name, attrs);
    else if ( xmlStrcasecmp(name, (xmlChar *) "triangle") == 0 )
      read_triangle(name, attrs);
    else if ( xmlStrcasecmp(name, (xmlChar *) "tetrahedron") == 0 )
      read_tetrahedron(name, attrs);
    
    break;

  case INSIDE_COORDINATES:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "array") == 0 )
    {
      read_mesh_coord(name, attrs);
      state = INSIDE_VECTOR;
    }
    if ( xmlStrcasecmp(name, (xmlChar *) "finiteelement") == 0 )
    {
      read_fe_signature(name, attrs);
    }
    if ( xmlStrcasecmp(name, (xmlChar *) "dofmap") == 0 )
    {
      read_dof_map_signature(name, attrs);
    }

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void NewXMLMesh::end_element(const xmlChar *name)
{
  switch ( state )
  {
  case INSIDE_MESH:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "mesh") == 0 )
    {
      close_mesh();
      state = DONE;
      release();
    }
    
    break;
    
  case INSIDE_VERTICES:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "vertices") == 0 )
    {
      state = INSIDE_MESH;    
    }

    break;

  case INSIDE_CELLS:
	 
    if ( xmlStrcasecmp(name, (xmlChar *) "cells") == 0 )
    {
      state = INSIDE_MESH;
    }

    break;

  case INSIDE_COORDINATES:

    if ( xmlStrcasecmp(name, (xmlChar *) "coordinates") == 0 )
    {
      state = INSIDE_MESH;
    }

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void NewXMLMesh::write(const Mesh& mesh, std::ostream& outfile, uint indentation_level)
{
  uint curr_indent = indentation_level;

  // Get cell type
  CellType::Type cell_type = mesh.type().cellType();

  // Write mesh header
  outfile << std::setw(curr_indent) << "";
  outfile << "<mesh celltype=\"" << CellType::type2string(cell_type) << "\" dim=\"" << mesh.geometry().dim() << "\">" << std::endl;

  // Write vertices header
  curr_indent = indentation_level + 2;
  outfile << std::setw(curr_indent) << "";
  outfile << "<vertices size=\"" << mesh.numVertices() << "\">" << std::endl;

  // Write each vertex 
  curr_indent = indentation_level + 4;
  for(VertexIterator v(mesh); !v.end(); ++v)
  {
    Point p = v->point();
    outfile << std::setw(curr_indent) << "";

    switch ( mesh.geometry().dim() ) {
    case 1:
      outfile << "<vertex index=\"" << v->index() << "\" x=\"" << p.x() << "\"/>" << std::endl;
      break;
    case 2:
      outfile << "<vertex index=\"" << v->index() << "\" x=\"" << p.x() << "\" y=\"" << p.y() << "\"/>" << std::endl;
      break;
    case 3:
      outfile << "<vertex index=\"" << v->index() << "\" x=\"" << p.x() << "\" y=\"" << p.y()  << "\" z=\"" << p.z() << "\"/>" << std::endl;
      break;
    default:
      error("The XML mesh file format only supports 1D, 2D and 3D meshes.");
    }
  }

  // Write vertex footer
  curr_indent = indentation_level + 2;
  outfile << std::setw(curr_indent) << "";
  outfile << "</vertices>" << std::endl;

  // Write cell header
  outfile << std::setw(curr_indent) << "";
  outfile << "<cells size=\"" << mesh.numCells() << "\">" << std::endl;

  // Write each cell
  curr_indent = indentation_level + 4;
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    const uint* vertices = c->entities(0);
    dolfin_assert(vertices);
    outfile << std::setw(curr_indent) << "";

    switch ( cell_type )
    {
    case CellType::interval:
      outfile << "<interval index=\"" <<  c->index() << "\" v0=\"" << vertices[0] << "\" v1=\"" << vertices[1] << "\"/>" << std::endl;
      break;
    case CellType::triangle:
      outfile << "<triangle index=\"" <<  c->index() << "\" v0=\"" << vertices[0] << "\" v1=\"" << vertices[1] << "\" v2=\"" << vertices[2] << "\"/>" << std::endl;
      break;
    case CellType::tetrahedron:
      outfile << "<triangle index=\"" <<  c->index() << "\" v0=\"" << vertices[0] << "\" v1=\"" << vertices[1] << "\" v2=\"" << vertices[2] << "\" v3=\"" << vertices[3] << "\"/>" << std::endl;
      break;
    default:
      error("Unknown cell type: %u.", cell_type);
    }
  }
  // Write cell footer 
  curr_indent = indentation_level + 2;
  outfile << std::setw(curr_indent) << "";
  outfile << "</cells>" << std::endl;

  // Write mesh data
  XMLMeshData::write(mesh.data(), outfile, indentation_level + 2);

  // Write mesh footer 
  curr_indent = indentation_level;
  outfile << std::setw(curr_indent) << "";
  outfile << "</mesh>" << std::endl;

}
//-----------------------------------------------------------------------------
void NewXMLMesh::read_mesh_tag(const xmlChar *name, const xmlChar **attrs)
{
  // Set state
  state = INSIDE_MESH;

  // Parse values
  std::string type = parse_string(name, attrs, "celltype");
  uint gdim = parse_uint(name, attrs, "dim");
  
  // Create cell type to get topological dimension
  CellType* cell_type = CellType::create(type);
  uint tdim = cell_type->dim();
  delete cell_type;

  // Open mesh for editing
  editor.open(_mesh, CellType::string2type(type), tdim, gdim);
}
//-----------------------------------------------------------------------------
void NewXMLMesh::read_vertices(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  uint num_vertices = parse_uint(name, attrs, "size");

  // Set number of vertices
  editor.initVertices(num_vertices);
}
//-----------------------------------------------------------------------------
void NewXMLMesh::read_cells(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  uint num_cells = parse_uint(name, attrs, "size");

  // Set number of vertices
  editor.initCells(num_cells);
}
//-----------------------------------------------------------------------------
void NewXMLMesh::read_vertex(const xmlChar *name, const xmlChar **attrs)
{
  // Read index
  uint v = parse_uint(name, attrs, "index");
  
  // Handle differently depending on geometric dimension
  switch ( _mesh.geometry().dim() )
  {
  case 1:
    {
      double x = parse_float(name, attrs, "x");
      editor.addVertex(v, x);
    }
    break;
  case 2:
    {
      double x = parse_float(name, attrs, "x");
      double y = parse_float(name, attrs, "y");
      editor.addVertex(v, x, y);
    }
    break;
  case 3:
    {
      double x = parse_float(name, attrs, "x");
      double y = parse_float(name, attrs, "y");
      double z = parse_float(name, attrs, "z");
      editor.addVertex(v, x, y, z);
    }
    break;
  default:
    error("Dimension of mesh must be 1, 2 or 3.");
  }
}
//-----------------------------------------------------------------------------
void NewXMLMesh::read_interval(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if ( _mesh.topology().dim() != 1 )
    error("Mesh entity (interval) does not match dimension of mesh (%d).",
		 _mesh.topology().dim());

  // Parse values
  uint c  = parse_uint(name, attrs, "index");
  uint v0 = parse_uint(name, attrs, "v0");
  uint v1 = parse_uint(name, attrs, "v1");
  
  // Add cell
  editor.addCell(c, v0, v1);
  
  // set affine indicator
  const std::string affine_str = parse_string_optional(name, attrs, "affine");
  editor.setAffineCellIndicator(c, affine_str);
}
//-----------------------------------------------------------------------------
void NewXMLMesh::read_triangle(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if ( _mesh.topology().dim() != 2 )
    error("Mesh entity (triangle) does not match dimension of mesh (%d).",
		 _mesh.topology().dim());

  // Parse values
  uint c  = parse_uint(name, attrs, "index");
  uint v0 = parse_uint(name, attrs, "v0");
  uint v1 = parse_uint(name, attrs, "v1");
  uint v2 = parse_uint(name, attrs, "v2");
  
  // Add cell
  editor.addCell(c, v0, v1, v2);
  
  // set affine indicator
  const std::string affine_str = parse_string_optional(name, attrs, "affine");
  editor.setAffineCellIndicator(c, affine_str);
}
//-----------------------------------------------------------------------------
void NewXMLMesh::read_tetrahedron(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if ( _mesh.topology().dim() != 3 )
    error("Mesh entity (tetrahedron) does not match dimension of mesh (%d).",
		 _mesh.topology().dim());

  // Parse values
  uint c  = parse_uint(name, attrs, "index");
  uint v0 = parse_uint(name, attrs, "v0");
  uint v1 = parse_uint(name, attrs, "v1");
  uint v2 = parse_uint(name, attrs, "v2");
  uint v3 = parse_uint(name, attrs, "v3");
  
  // Add cell
  editor.addCell(c, v0, v1, v2, v3);
  
  // set affine indicator
  const std::string affine_str = parse_string_optional(name, attrs, "affine");
  editor.setAffineCellIndicator(c, affine_str);
}
//-----------------------------------------------------------------------------
void NewXMLMesh::read_mesh_coord(const xmlChar* name, const xmlChar** attrs)
{
  delete xml_vector;
  mesh_coord = new Vector();
  xml_vector = new NewXMLVector(*mesh_coord, parser);
  xml_vector->handle();
}
//-----------------------------------------------------------------------------
void NewXMLMesh::read_mesh_entity(const xmlChar* name, const xmlChar** attrs)
{
  // Read index
  const uint index = parse_uint(name, attrs, "index");

  // Read and set value
  dolfin_assert(f);
  dolfin_assert(index < f->size());
  const uint value = parse_uint(name, attrs, "value");
  f->set(index, value);
}
//-----------------------------------------------------------------------------
void NewXMLMesh::read_mesh_data(const xmlChar* name, const xmlChar** attrs)
{
  xml_mesh_data = new XMLMeshData(_mesh.data(), parser, true);
  xml_mesh_data->handle();
}
//-----------------------------------------------------------------------------
void NewXMLMesh::read_fe_signature(const xmlChar* name, const xmlChar** attrs)
{
  // Read string for the finite element signature
  const std::string FE_signature = parse_string_optional(name, attrs, "signature");
  editor.setMeshCoordFEsignature(FE_signature);
}
//-----------------------------------------------------------------------------
void NewXMLMesh::read_dof_map_signature(const xmlChar* name, const xmlChar** attrs)
{
  // Read string for the dofmap signature
  const std::string dofmap_signature = parse_string_optional(name, attrs, "signature");
  editor.setMeshCoordDofMapsignature(dofmap_signature);
}
//-----------------------------------------------------------------------------
void NewXMLMesh::close_mesh()
{
  // Setup higher order mesh coordinate data
  // FIXME: This will introduce a memory leak
  //error("NewXMLMesh::closeMesh() introduces a memory leak. Please fix.");

  if (!mesh_coord == 0)
  {
    editor.setMeshCoordinates(*mesh_coord); 
    delete mesh_coord;
    delete xml_vector;
    mesh_coord = 0;
    xml_vector = 0;
  }
  editor.close(false);
}
//-----------------------------------------------------------------------------

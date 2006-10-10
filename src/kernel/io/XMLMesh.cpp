// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-10-21
// Last changed: 2006-06-22

#include <dolfin/dolfin_log.h>
#include <dolfin/CellType.h>
#include <dolfin/Mesh.h>
#include <dolfin/XMLMesh.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLMesh::XMLMesh(Mesh& mesh) : XMLObject(), _mesh(mesh), state(OUTSIDE)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLMesh::~XMLMesh()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLMesh::startElement(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state )
  {
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "mesh") == 0 )
    {
      readMesh(name, attrs);
      state = INSIDE_MESH;
    }
    
    break;

  case INSIDE_MESH:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "vertices") == 0 )
    {
      readVertices(name, attrs);
      state = INSIDE_VERTICES;
    }
    else if ( xmlStrcasecmp(name, (xmlChar *) "cells") == 0 )
    {
      readCells(name, attrs);
      state = INSIDE_CELLS;
    }
    
    break;
    
  case INSIDE_VERTICES:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "vertex") == 0 )
      readVertex(name, attrs);

    break;
    
  case INSIDE_CELLS:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "interval") == 0 )
      readInterval(name, attrs);
    else if ( xmlStrcasecmp(name, (xmlChar *) "triangle") == 0 )
      readTriangle(name, attrs);
    else if ( xmlStrcasecmp(name, (xmlChar *) "tetrahedron") == 0 )
      readTetrahedron(name, attrs);
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::endElement(const xmlChar *name)
{
  switch ( state )
  {
  case INSIDE_MESH:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "mesh") == 0 )
    {
      closeMesh();
      state = DONE;
    }
    
    break;
    
  case INSIDE_VERTICES:
    
    if ( xmlStrcasecmp(name, (xmlChar *) "vertices") == 0 )
      state = INSIDE_MESH;
    
    break;

  case INSIDE_CELLS:
	 
    if ( xmlStrcasecmp(name, (xmlChar *) "cells") == 0 )
      state = INSIDE_MESH;
    
    break;
    
  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::open(std::string filename)
{
  cout << "Reading mesh from file \"" << filename << "\"." << endl;
}
//-----------------------------------------------------------------------------
bool XMLMesh::close()
{
  return state == DONE;
}
//-----------------------------------------------------------------------------
void XMLMesh::readMesh(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  uint dim = parseUnsignedInt(name, attrs, "dim");
  std::string cell_type = parseString(name, attrs, "celltype");
  
  // Check values
  if ( dim < 1 || dim > 3 )
    dolfin_error("Dimension of mesh must be 1, 2 or 3.");

  // Open mesh for editing
  editor.open(_mesh, CellType::type(cell_type), dim, dim);
}
//-----------------------------------------------------------------------------
void XMLMesh::readVertices(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  uint num_vertices = parseUnsignedInt(name, attrs, "size");

  // Set number of vertices
  editor.initVertices(num_vertices);
}
//-----------------------------------------------------------------------------
void XMLMesh::readCells(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  uint num_cells = parseUnsignedInt(name, attrs, "size");

  // Set number of vertices
  editor.initCells(num_cells);
}
//-----------------------------------------------------------------------------
void XMLMesh::readVertex(const xmlChar *name, const xmlChar **attrs)
{
  // Read index
  uint v = parseUnsignedInt(name, attrs, "index");
  
  // Handle differently depending on dimension
  switch ( _mesh.dim() )
  {
  case 1:
    {
      real x = parseReal(name, attrs, "x");
      editor.addVertex(v, x);
    }
    break;
  case 2:
    {
      real x = parseReal(name, attrs, "x");
      real y = parseReal(name, attrs, "y");
      editor.addVertex(v, x, y);
    }
    break;
  case 3:
    {
      real x = parseReal(name, attrs, "x");
      real y = parseReal(name, attrs, "y");
      real z = parseReal(name, attrs, "z");
      editor.addVertex(v, x, y, z);
    }
    break;
  default:
    dolfin_error("Dimension of mesh must be 1, 2 or 3.");
  }
}
//-----------------------------------------------------------------------------
void XMLMesh::readInterval(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if ( _mesh.dim() != 1 )
    dolfin_error1("Mesh entity (interval) does not match dimension of mesh (%d).",
		 _mesh.dim());

  // Parse values
  uint c  = parseUnsignedInt(name, attrs, "index");
  uint v0 = parseUnsignedInt(name, attrs, "v0");
  uint v1 = parseUnsignedInt(name, attrs, "v1");
  
  // Add cell
  editor.addCell(c, v0, v1);
}
//-----------------------------------------------------------------------------
void XMLMesh::readTriangle(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if ( _mesh.dim() != 2 )
    dolfin_error1("Mesh entity (triangle) does not match dimension of mesh (%d).",
		 _mesh.dim());

  // Parse values
  uint c  = parseUnsignedInt(name, attrs, "index");
  uint v0 = parseUnsignedInt(name, attrs, "v0");
  uint v1 = parseUnsignedInt(name, attrs, "v1");
  uint v2 = parseUnsignedInt(name, attrs, "v2");
  
  // Add cell
  editor.addCell(c, v0, v1, v2);
}
//-----------------------------------------------------------------------------
void XMLMesh::readTetrahedron(const xmlChar *name, const xmlChar **attrs)
{
  // Check dimension
  if ( _mesh.dim() != 3 )
    dolfin_error1("Mesh entity (tetrahedron) does not match dimension of mesh (%d).",
		 _mesh.dim());

  // Parse values
  uint c  = parseUnsignedInt(name, attrs, "index");
  uint v0 = parseUnsignedInt(name, attrs, "v0");
  uint v1 = parseUnsignedInt(name, attrs, "v1");
  uint v2 = parseUnsignedInt(name, attrs, "v2");
  uint v3 = parseUnsignedInt(name, attrs, "v3");
  
  // Add cell
  editor.addCell(c, v0, v1, v2, v3);
}
//-----------------------------------------------------------------------------
void XMLMesh::closeMesh()
{
  editor.close();
}
//-----------------------------------------------------------------------------

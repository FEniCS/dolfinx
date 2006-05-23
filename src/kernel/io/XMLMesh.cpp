// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-10-21
// Last changed: 2006-05-23

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/Cell.h>
#include <dolfin/MeshData.h>
#include <dolfin/XMLMesh.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLMesh::XMLMesh(Mesh& mesh_) : XMLObject(), mesh(mesh_)
{
  state = OUTSIDE;
  vertices = 0;
  cells = 0;
}
//-----------------------------------------------------------------------------
void XMLMesh::startElement(const xmlChar *name, const xmlChar **attrs)
{
  //dolfin_debug1("Found start of element \"%s\"", (const char *) name);

  switch ( state ){
  case OUTSIDE:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "mesh") == 0 ) {
      readMesh(name,attrs);
      state = INSIDE_MESH;
    }
    
    break;

  case INSIDE_MESH:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "vertices") == 0 ) {
      readVertices(name,attrs);
      state = INSIDE_VERTICES;
    }
    else if ( xmlStrcasecmp(name,(xmlChar *) "cells") == 0 ) {
      readCells(name,attrs);
      state = INSIDE_CELLS;
    }
    
    break;
    
  case INSIDE_VERTICES:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "vertex") == 0 )
    {
      readVertex(name,attrs);
      state = INSIDE_VERTEX;
    }

    break;

  case INSIDE_VERTEX:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "boundaryid") == 0 )
      readBoundaryID(name,attrs);
    
    break;
    
  case INSIDE_CELLS:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "triangle") == 0 )
      readTriangle(name,attrs);
    if ( xmlStrcasecmp(name,(xmlChar *) "tetrahedron") == 0 )
      readTetrahedron(name,attrs);
    
    break;
    
  default:
    ;
  }
  
}
//-----------------------------------------------------------------------------
void XMLMesh::endElement(const xmlChar *name)
{
  //dolfin_debug1("Found end of element \"%s\"", (const char *) name);

  switch ( state ){
  case INSIDE_MESH:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "mesh") == 0 ) {
      initMesh();
      state = DONE;
    }
    
    break;
    
  case INSIDE_VERTICES:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "vertices") == 0 )
      state = INSIDE_MESH;
    
    break;

  case INSIDE_VERTEX:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "vertex") == 0 )
      state = INSIDE_VERTICES;
    
    break;
    
  case INSIDE_CELLS:
	 
    if ( xmlStrcasecmp(name,(xmlChar *) "cells") == 0 )
      state = INSIDE_MESH;
    
    break;
    
  default:
    ;
  }
  
}
//-----------------------------------------------------------------------------
void XMLMesh::reading(std::string filename)
{
  cout << "Reading mesh from file \"" << filename << "\"." << endl;
}
//-----------------------------------------------------------------------------
void XMLMesh::done()
{
  //cout << "Reading mesh: " << mesh << endl;
}
//-----------------------------------------------------------------------------
void XMLMesh::readMesh(const xmlChar *name, const xmlChar **attrs)
{
  mesh.clear();
}
//-----------------------------------------------------------------------------
void XMLMesh::readVertices(const xmlChar *name, const xmlChar **attrs)
{
  vertices = parseInt(name, attrs, "size");
}
//-----------------------------------------------------------------------------
void XMLMesh::readCells(const xmlChar *name, const xmlChar **attrs)
{
  cells = parseInt(name, attrs, "size");
}
//-----------------------------------------------------------------------------
void XMLMesh::readVertex(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  //int id  = parseInt(name, attrs, "name");
  real x  = parseReal(name, attrs, "x");
  real y  = parseReal(name, attrs, "y");
  real z  = parseReal(name, attrs, "z");

  // Set values
  Vertex &newvertex = mesh.createVertex(x, y, z);
  vertex = &newvertex;

  // FIXME: id of vertex is completely ignored. We assume that the
  // vertices are in correct order.
}
//-----------------------------------------------------------------------------
void XMLMesh::readBoundaryID(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  int id = parseInt(name, attrs, "name");

  // Add Boundary ID to vertex
  vertex->nbids.insert(id);

  // FIXME: id of vertex is completely ignored. We assume that the
  // vertices are in correct order.
}
//-----------------------------------------------------------------------------
void XMLMesh::readTriangle(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  //int id = parseInt(name, attrs, "name");
  int n0 = parseInt(name, attrs, "n0");
  int n1 = parseInt(name, attrs, "n1");
  int n2 = parseInt(name, attrs, "n2");

  // Set values
  mesh.createCell(n0, n1, n2);

  // FIXME: id of cell is completely ignored. We assume that the
  // cells are in correct order.
}
//-----------------------------------------------------------------------------
void XMLMesh::readTetrahedron(const xmlChar *name, const xmlChar **attrs)
{
  // Parse values
  //int id = parseIntegerRequired(name, attrs, "name");
  int n0 = parseInt(name, attrs, "n0");
  int n1 = parseInt(name, attrs, "n1");
  int n2 = parseInt(name, attrs, "n2");
  int n3 = parseInt(name, attrs, "n3");

  // Set values
  mesh.createCell(n0, n1, n2, n3);

  // FIXME: id of cell is completely ignored. We assume that the
  // cells are in correct order.
}
//-----------------------------------------------------------------------------
void XMLMesh::initMesh()
{
  // Compute connections
  mesh.init();
}
//-----------------------------------------------------------------------------

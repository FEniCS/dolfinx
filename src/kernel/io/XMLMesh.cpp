// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Mesh.h>
#include <dolfin/Cell.h>
#include <dolfin/MeshData.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/XMLMesh.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLMesh::XMLMesh(Mesh& mesh_) : XMLObject(), mesh(mesh_)
{
  state = OUTSIDE;
  nodes = 0;
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
    
    if ( xmlStrcasecmp(name,(xmlChar *) "nodes") == 0 ) {
      readNodes(name,attrs);
      state = INSIDE_NODES;
    }
    else if ( xmlStrcasecmp(name,(xmlChar *) "cells") == 0 ) {
      readCells(name,attrs);
      state = INSIDE_CELLS;
    }
    
    break;
    
  case INSIDE_NODES:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "node") == 0 )
      readNode(name,attrs);
    
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
      ok = true;
      state = DONE;
    }
    
    break;
    
  case INSIDE_NODES:
    
    if ( xmlStrcasecmp(name,(xmlChar *) "nodes") == 0 )
      state = INSIDE_MESH;
    
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
void XMLMesh::readNodes(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  int size = 0;

  // Parse values
  parseIntegerRequired(name, attrs, "size", &size);

  // Set values
  nodes = size;
}
//-----------------------------------------------------------------------------
void XMLMesh::readCells(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  int size = 0;
  
  // Parse values
  parseIntegerRequired(name, attrs, "size", &size);

  // Set values
  cells = size;
}
//-----------------------------------------------------------------------------
void XMLMesh::readNode(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  int id = 0;
  real x = 0.0;
  real y = 0.0;
  real z = 0.0;
  
  // Parse values
  parseIntegerRequired(name, attrs, "name", &id);
  parseRealRequired(name, attrs, "x", &x);
  parseRealRequired(name, attrs, "y", &y);
  parseRealRequired(name, attrs, "z", &z);

  // Set values
  mesh.createNode(x, y, z);

  // FIXME: id of node is completely ignored. We assume that the
  // nodes are in correct order.
}
//-----------------------------------------------------------------------------
void XMLMesh::readTriangle(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  int id = 0;
  int n0 = 0;
  int n1 = 0;
  int n2 = 0;
  
  // Parse values
  parseIntegerRequired(name, attrs, "name", &id);
  parseIntegerRequired(name, attrs, "n0", &n0);
  parseIntegerRequired(name, attrs, "n1", &n1);
  parseIntegerRequired(name, attrs, "n2", &n2);

  // Set values
  mesh.createCell(n0, n1, n2);

  // FIXME: id of cell is completely ignored. We assume that the
  // cells are in correct order.
}
//-----------------------------------------------------------------------------
void XMLMesh::readTetrahedron(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  int id = 0;
  int n0 = 0;
  int n1 = 0;
  int n2 = 0;
  int n3 = 0;
  
  // Parse values
  parseIntegerRequired(name, attrs, "name", &id);
  parseIntegerRequired(name, attrs, "n0", &n0);
  parseIntegerRequired(name, attrs, "n1", &n1);
  parseIntegerRequired(name, attrs, "n2", &n2);
  parseIntegerRequired(name, attrs, "n3", &n3);

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

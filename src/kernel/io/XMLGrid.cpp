// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Grid.h>
#include "GridData.h"
#include "XMLGrid.h"

// Fixme: temporary
#include <iostream>

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLGrid::XMLGrid(Grid *grid) : XMLObject()
{
  this->grid = grid;
  state = OUTSIDE;
  nodes = 0;
  cells = 0;
}
//-----------------------------------------------------------------------------
void XMLGrid::startElement(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state ){
  case OUTSIDE:

	 if ( xmlStrcasecmp(name,(xmlChar *) "grid") == 0 ){
		readGrid(name,attrs);
		state = INSIDE_GRID;
	 }
	 
	 break;
  case INSIDE_GRID:
	 
	 if ( xmlStrcasecmp(name,(xmlChar *) "nodes") == 0 ){
		readNodes(name,attrs);
		state = INSIDE_NODES;
	 }
	 else if ( xmlStrcasecmp(name,(xmlChar *) "cells") == 0 ){
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
  }
  
}
//-----------------------------------------------------------------------------
void XMLGrid::endElement(const xmlChar *name)
{
  switch ( state ){
  case INSIDE_GRID:
	 
	 if ( xmlStrcasecmp(name,(xmlChar *) "grid") == 0 ){
		ok = true;
		state = DONE;
	 }
	 
	 break;
  case INSIDE_NODES:

	 if ( xmlStrcasecmp(name,(xmlChar *) "nodes") == 0 )
		state = INSIDE_GRID;
	 
	 break;
  case INSIDE_CELLS:
	 
	 if ( xmlStrcasecmp(name,(xmlChar *) "cells") == 0 )
		state = INSIDE_GRID;
	 
	 break;
  }

}
//-----------------------------------------------------------------------------
void XMLGrid::readGrid(const xmlChar *name, const xmlChar **attrs)
{
  // Clear grid
  grid->clear();
}
//-----------------------------------------------------------------------------
void XMLGrid::readNodes(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  int size = 0;

  // Parse values
  parseIntegerRequired(name, attrs, "size", &size);

  // Set values
  nodes = size;
}
//-----------------------------------------------------------------------------
void XMLGrid::readCells(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  int size = 0;
  
  // Parse values
  parseIntegerRequired(name, attrs, "size", &size);

  // Set values
  cells = size;
}
//-----------------------------------------------------------------------------
void XMLGrid::readNode(const xmlChar *name, const xmlChar **attrs)
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
  grid->gridData->createNode(x,y,z);

  // FIXME: id of node is completely ignored. We assume that the
  // nodes are in correct order.
}
//-----------------------------------------------------------------------------
void XMLGrid::readTriangle(const xmlChar *name, const xmlChar **attrs)
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
  grid->gridData->createTriangle(n0,n1,n2);

  // FIXME: id of cell is completely ignored. We assume that the
  // cells are in correct order.
}
//-----------------------------------------------------------------------------
void XMLGrid::readTetrahedron(const xmlChar *name, const xmlChar **attrs)
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
  grid->gridData->createTetrahedron(n0,n1,n2,n3);

  // FIXME: id of cell is completely ignored. We assume that the
  // cells are in correct order.
}
//-----------------------------------------------------------------------------

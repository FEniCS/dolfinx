// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Grid.h>
#include "XMLGrid.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLGrid::XMLGrid(Grid *grid) : XMLObject()
{
  this->grid = grid;
  state = OUTSIDE;
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
	 
	 if ( xmlStrcasecmp(name,(xmlChar *) "cell") == 0 )
		readCell(name,attrs);
	 
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
  // Set default values
  int rows = 0;
  int columns = 0;

  // Parse values
  parseIntegerRequired(name, attrs, "rows",    &rows);
  parseIntegerRequired(name, attrs, "columns", &columns);

  // Set values

}
//-----------------------------------------------------------------------------
void XMLGrid::readNodes(const xmlChar *name, const xmlChar **attrs)
{


}
//-----------------------------------------------------------------------------
void XMLGrid::readCells(const xmlChar *name, const xmlChar **attrs)
{


}
//-----------------------------------------------------------------------------
void XMLGrid::readNode(const xmlChar *name, const xmlChar **attrs)
{


}
//-----------------------------------------------------------------------------
void XMLGrid::readCell(const xmlChar *name, const xmlChar **attrs)
{


}
//-----------------------------------------------------------------------------

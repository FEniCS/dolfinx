// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __XML_GRID_H
#define __XML_GRID_H

#include "XMLObject.h"

namespace dolfin {

  class Grid;
  
  class XMLGrid : public XMLObject {
  public:

	 XMLGrid(Grid *grid);
	 
	 void startElement (const xmlChar *name, const xmlChar **attrs);
	 void endElement   (const xmlChar *name);
	 
  private:

	 enum ParserState { OUTSIDE, INSIDE_GRID, INSIDE_NODES, INSIDE_CELLS, DONE };
	 
	 void readGrid  (const xmlChar *name, const xmlChar **attrs);
	 void readNodes (const xmlChar *name, const xmlChar **attrs);
	 void readCells (const xmlChar *name, const xmlChar **attrs);
	 void readNode  (const xmlChar *name, const xmlChar **attrs);
	 void readCell  (const xmlChar *name, const xmlChar **attrs);

	 Grid *grid;

	 ParserState state;
	 
  };
  
}

#endif

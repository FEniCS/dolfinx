// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-10-21
// Last changed: 2005-12-01

#ifndef __XML_MESH_H
#define __XML_MESH_H

#include <dolfin/XMLObject.h>

namespace dolfin
{
  class Mesh;
  
  class XMLMesh : public XMLObject
  {
  public:

    XMLMesh(Mesh& mesh_);
    
    void startElement (const xmlChar *name, const xmlChar **attrs);
    void endElement   (const xmlChar *name);
    
    void reading(std::string filename);
    void done();
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_MESH, INSIDE_VERTICES, INSIDE_VERTEX, INSIDE_CELLS, DONE };
    
    void readMesh        (const xmlChar *name, const xmlChar **attrs);
    void readVertices    (const xmlChar *name, const xmlChar **attrs);
    void readCells       (const xmlChar *name, const xmlChar **attrs);
    void readVertex      (const xmlChar *name, const xmlChar **attrs);
    void readBoundaryID  (const xmlChar *name, const xmlChar **attrs);
    void readTriangle    (const xmlChar *name, const xmlChar **attrs);
    void readTetrahedron (const xmlChar *name, const xmlChar **attrs);
    
    void initMesh();

    Mesh& mesh;
    int vertices;
    int cells;
    Vertex *vertex;
    
    bool _create_edges;
    
    ParserState state;
    
  };
  
}

#endif

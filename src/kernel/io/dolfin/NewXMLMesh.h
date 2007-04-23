// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-10-21
// Last changed: 2006-05-23

#ifndef __NEW_XML_MESH_H
#define __NEW_XML_MESH_H

#include <dolfin/MeshEditor.h>
#include <dolfin/XMLObject.h>

namespace dolfin
{
  
  class NewMesh;
  
  class NewXMLMesh : public XMLObject
  {
  public:

    NewXMLMesh(NewMesh& mesh);
    ~NewXMLMesh();
    
    void startElement (const xmlChar* name, const xmlChar** attrs);
    void endElement   (const xmlChar* name);
    
    void open(std::string filename);
    bool close();
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_MESH, INSIDE_VERTICES, INSIDE_CELLS, DONE };
    
    void readMesh        (const xmlChar* name, const xmlChar** attrs);
    void readVertices    (const xmlChar* name, const xmlChar** attrs);
    void readCells       (const xmlChar* name, const xmlChar** attrs);
    void readVertex      (const xmlChar* name, const xmlChar** attrs);
    void readInterval    (const xmlChar* name, const xmlChar** attrs);
    void readTriangle    (const xmlChar* name, const xmlChar** attrs);
    void readTetrahedron (const xmlChar* name, const xmlChar** attrs);
    
    void closeMesh();

    NewMesh& _mesh;
    ParserState state;
    MeshEditor editor;
    
  };
  
}

#endif

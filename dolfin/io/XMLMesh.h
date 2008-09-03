// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-10-21
// Last changed: 2008-05-21

#ifndef __NEW_XML_MESH_H
#define __NEW_XML_MESH_H

#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshFunction.h>
#include "XMLObject.h"

#include "XMLVector.h"
#include <dolfin/la/Vector.h>

namespace dolfin
{
  
  class Mesh;
  
  class XMLMesh : public XMLObject
  {
  public:

    XMLMesh(Mesh& mesh);
    ~XMLMesh();
    
    void startElement (const xmlChar* name, const xmlChar** attrs);
    void endElement   (const xmlChar* name);
    
    void open(std::string filename);
    bool close();
    
  private:
    
    enum ParserState {OUTSIDE,
                      INSIDE_MESH, INSIDE_VERTICES, INSIDE_CELLS, INSIDE_COORDINATES,
                      INSIDE_DATA, INSIDE_MESH_FUNCTION, INSIDE_ARRAY, INSIDE_VECTOR,
                      DONE};
    
    void readMesh        (const xmlChar* name, const xmlChar** attrs);
    void readVertices    (const xmlChar* name, const xmlChar** attrs);
    void readCells       (const xmlChar* name, const xmlChar** attrs);
    void readCoordinates (const xmlChar* name, const xmlChar** attrs);
    void readVertex      (const xmlChar* name, const xmlChar** attrs);
    void readInterval    (const xmlChar* name, const xmlChar** attrs);
    void readTriangle    (const xmlChar* name, const xmlChar** attrs);
    void readTetrahedron (const xmlChar* name, const xmlChar** attrs);
    void readMeshFunction(const xmlChar* name, const xmlChar** attrs);
    void readArray       (const xmlChar* name, const xmlChar** attrs);
    void readMeshEntity  (const xmlChar* name, const xmlChar** attrs);
    void readArrayElement(const xmlChar* name, const xmlChar** attrs);

    void readFEsignature    (const xmlChar* name, const xmlChar** attrs);
    void readDofMapsignature(const xmlChar* name, const xmlChar** attrs);
    
    void closeMesh();

    Mesh& _mesh;
    ParserState state;
    MeshEditor editor;
    MeshFunction<uint>* f;
    Array<uint>* a;
    
    // variables for reading in higher order mesh coordinates from vector data
    Vector *mesh_coord_vec;
    XMLVector *xml_vec;
    
  };
  
}

#endif

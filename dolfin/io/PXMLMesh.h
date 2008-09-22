// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-10-21
// Last changed: 2008-05-21

#ifndef __NEW_PXML_MESH_H
#define __NEW_PXML_MESH_H

#include <dolfin/common/Array.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshFunction.h>
#include "XMLObject.h"
#include <map>

namespace dolfin
{
  
  class Mesh;
  
  class PXMLMesh : public XMLObject
  {
  public:

    PXMLMesh(Mesh& mesh);
    ~PXMLMesh();
    
    void startElement (const xmlChar* name, const xmlChar** attrs);
    void endElement   (const xmlChar* name);
    
    void open(std::string filename);
    bool close();
    
  private:

    // Possible parser states
    enum ParserState {OUTSIDE,
                      INSIDE_MESH, INSIDE_VERTICES, INSIDE_CELLS,
                      INSIDE_DATA, INSIDE_MESH_FUNCTION, INSIDE_ARRAY,
                      DONE};
    
    // Callbacks for reading XML data
    void readMesh        (const xmlChar* name, const xmlChar** attrs);
    void readVertices    (const xmlChar* name, const xmlChar** attrs);
    void readCells       (const xmlChar* name, const xmlChar** attrs);
    void readVertex      (const xmlChar* name, const xmlChar** attrs);
    void readInterval    (const xmlChar* name, const xmlChar** attrs);
    void readTriangle    (const xmlChar* name, const xmlChar** attrs);
    void readTetrahedron (const xmlChar* name, const xmlChar** attrs);
    void readMeshFunction(const xmlChar* name, const xmlChar** attrs);
    void readArray       (const xmlChar* name, const xmlChar** attrs);
    void readMeshEntity  (const xmlChar* name, const xmlChar** attrs);
    void readArrayElement(const xmlChar* name, const xmlChar** attrs);
    
    // Close mesh, called when finished reading data
    void closeMesh();

    // Reference to the mesh
    Mesh& _mesh;

    // Parser state
    ParserState state;

    // Mesh editor
    MeshEditor editor;

    // Pointer to current mesh function, used when reading mesh function data
    MeshFunction<uint>* f;

    // Pointer to current array, used when reading array data
    Array<uint>* a;

    // FIXME: Initialize all these in constructor

    // Data for parallel parsing
    std::map<uint, uint>* local_to_global;    
    uint start_index, end_index, num_parsed_v;
    MeshFunction<uint>* global_numbering;
    Array<uint> cell_buffer;

    // FIXME replace these with hash tables
    std::set<uint> local_vertex, shared_vertex, used_vertex;
    std::set<uint>::iterator it;

  };
  
}

#endif

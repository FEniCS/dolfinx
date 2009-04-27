// Copyright (C) 2003-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-10-21
// Last changed: 2009-02-26

#ifndef __XML_MESH_H
#define __XML_MESH_H

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
    
    void start_element (const xmlChar* name, const xmlChar** attrs);
    void end_element   (const xmlChar* name);
    
    void open(std::string filename);
    bool close();
    
  private:
    
    enum ParserState {OUTSIDE,
                      INSIDE_MESH, INSIDE_VERTICES, INSIDE_CELLS, INSIDE_COORDINATES,
                      INSIDE_DATA, INSIDE_MESH_FUNCTION, INSIDE_ARRAY, INSIDE_VECTOR,
                      DONE};
    
    void read_mesh        (const xmlChar* name, const xmlChar** attrs);
    void read_vertices    (const xmlChar* name, const xmlChar** attrs);
    void read_cells       (const xmlChar* name, const xmlChar** attrs);
    void read_vertex      (const xmlChar* name, const xmlChar** attrs);
    void read_interval    (const xmlChar* name, const xmlChar** attrs);
    void read_triangle    (const xmlChar* name, const xmlChar** attrs);
    void read_tetrahedron (const xmlChar* name, const xmlChar** attrs);
    void read_meshFunction(const xmlChar* name, const xmlChar** attrs);
    void read_array       (const xmlChar* name, const xmlChar** attrs);
    void read_meshEntity  (const xmlChar* name, const xmlChar** attrs);
    void read_arrayElement(const xmlChar* name, const xmlChar** attrs);

    void close_mesh();

    Mesh& _mesh;
    ParserState state;
    MeshEditor editor;
    MeshFunction<uint>* f;
    std::vector<uint>* a;
    
  };
  
}

#endif

// Copyright (C) 2009 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-06
// Last changed:  2009-03-11

#ifndef __NEW_XML_MESH_H
#define __NEW_XML_MESH_H

#include <dolfin/la/Vector.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshFunction.h>
#include "NewXMLVector.h"
#include "XMLHandler.h"

namespace dolfin
{
  
  class Mesh;
  class XMLMeshData;
  
  class NewXMLMesh : public XMLHandler
  {
  public:

    NewXMLMesh(Mesh& mesh, NewXMLFile& parser);
    ~NewXMLMesh();
    
    void start_element (const xmlChar* name, const xmlChar** attrs);
    void end_element   (const xmlChar* name);

    static void write(const Mesh& mesh, std::ostream& outfile, uint indentation_level=0);
    
  private:
    
    enum parser_state {OUTSIDE, INSIDE_MESH, INSIDE_VERTICES, INSIDE_CELLS, 
                       INSIDE_COORDINATES, INSIDE_VECTOR, DONE};
    
    void read_mesh          (const xmlChar* name, const xmlChar** attrs);
    void read_vertices      (const xmlChar* name, const xmlChar** attrs);
    void read_cells         (const xmlChar* name, const xmlChar** attrs);
    void read_coordinates   (const xmlChar* name, const xmlChar** attrs);
    void read_vertex        (const xmlChar* name, const xmlChar** attrs);
    void read_interval      (const xmlChar* name, const xmlChar** attrs);
    void read_triangle      (const xmlChar* name, const xmlChar** attrs);
    void read_tetrahedron   (const xmlChar* name, const xmlChar** attrs);
    void read_mesh_coord    (const xmlChar* name, const xmlChar** attrs);
    void read_mesh_entity   (const xmlChar* name, const xmlChar** attrs);
    void read_mesh_data     (const xmlChar* name, const xmlChar** attrs);

    void read_fe_signature  (const xmlChar* name, const xmlChar** attrs);
    void read_dof_map_signature(const xmlChar* name, const xmlChar** attrs);
    
    void close_mesh();

    Mesh& _mesh;
    parser_state state;
    MeshEditor editor;

    MeshFunction<uint>* f;
    std::vector<uint>* a;
    
    // Variables for reading in higher order mesh coordinates from vector data
    Vector* mesh_coord;
    std::vector<uint>* uint_array;
    XMLArray *xml_array;
    NewXMLVector* xml_vector;
    XMLMeshData* xml_mesh_data;
    
  };
  
}

#endif

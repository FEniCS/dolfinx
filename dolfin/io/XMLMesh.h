// Copyright (C) 2009 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-06
// Last changed:  2009-03-16

#ifndef __XML_MESH_H
#define __XML_MESH_H

#include <boost/scoped_ptr.hpp>
#include <dolfin/la/Vector.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/MeshFunction.h>
#include "XMLVector.h"
#include "XMLHandler.h"

namespace dolfin
{

  class Mesh;
  class XMLMeshData;

  class XMLMesh : public XMLHandler
  {
  public:

    XMLMesh(Mesh& mesh, XMLFile& parser);
    XMLMesh(Mesh& mesh, XMLFile& parser, std::string celltype, uint dim);
    ~XMLMesh();

    void start_element (const xmlChar* name, const xmlChar** attrs);
    void end_element   (const xmlChar* name);

    static void write(const Mesh& mesh, std::ostream& outfile, uint indentation_level=0);

    void read_mesh_tag(const xmlChar* name, const xmlChar** attrs);

  private:

    enum parser_state {OUTSIDE, INSIDE_MESH, INSIDE_VERTICES, INSIDE_HIGHERORDERVERTICES, INSIDE_CELLS,
                       INSIDE_HIGHERORDERCELLS, INSIDE_VECTOR, DONE};

    void read_vertices                  (const xmlChar* name, const xmlChar** attrs);
    void read_cells                     (const xmlChar* name, const xmlChar** attrs);
    void read_higher_order_vertices     (const xmlChar* name, const xmlChar** attrs);
    void read_higher_order_cells        (const xmlChar* name, const xmlChar** attrs);
    void read_vertex                    (const xmlChar* name, const xmlChar** attrs);
    void read_interval                  (const xmlChar* name, const xmlChar** attrs);
    void read_triangle                  (const xmlChar* name, const xmlChar** attrs);
    void read_tetrahedron               (const xmlChar* name, const xmlChar** attrs);
    void read_higher_order_vertex       (const xmlChar* name, const xmlChar** attrs);
    void read_higher_order_cell_data    (const xmlChar* name, const xmlChar** attrs);
    void read_mesh_entity               (const xmlChar* name, const xmlChar** attrs);
    void read_mesh_data                 (const xmlChar* name, const xmlChar** attrs);

    void close_mesh();

    Mesh& _mesh;
    parser_state state;
    MeshEditor editor;

    MeshFunction<uint>* f;
    std::vector<uint>* a;

    boost::scoped_ptr<XMLMeshData> xml_mesh_data;
  };

}

#endif

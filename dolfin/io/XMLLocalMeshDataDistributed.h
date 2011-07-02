// Copyright (C) 2006-2011 Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-03-10
// Last changed: 2011-05-30
//
// Modified by Kent-Andre Mardal, 2011.

#ifndef __XMLLOCALMESHDATADISTRIBUTED_H
#define __XMLLOCALMESHDATADISTRIBUTED_H

#include <string>
#include <libxml/parser.h>

namespace dolfin
{

  class LocalMeshData;

  /// Documentation of class XMLLocalMeshData

  class XMLLocalMeshDataDistributed
  {

  public:

    XMLLocalMeshDataDistributed(LocalMeshData& mesh_data,
                                const std::string filename);

    void read();

    void start_element(const xmlChar* name, const xmlChar** attrs, uint num_attributes);
    void end_element(const xmlChar* name);

  private:

    enum ParserState {OUTSIDE,
                      INSIDE_MESH, INSIDE_VERTICES, INSIDE_CELLS,
                      INSIDE_DATA, INSIDE_MESH_FUNCTION, INSIDE_ARRAY,
                      INSIDE_DATA_ENTRY,
                      DONE};

    static void sax_start_document(void *ctx);
    static void sax_end_document(void *ctx);

    //static void sax_start_element(void *ctx, const xmlChar *name, const xmlChar **attrs);
    //static void sax_end_element(void *ctx, const xmlChar *name);



    static void sax_start_element( void * ctx,
                               const xmlChar * name,
                               const xmlChar * prefix,
                               const xmlChar * URI,
                               int nb_namespaces,
                               const xmlChar ** namespaces,
                               int nb_attributes,
                               int nb_defaulted,
                               const xmlChar ** attrs);

    static void sax_end_element(void * ctx,
					 const xmlChar * name,
					 const xmlChar * prefix,
					 const xmlChar * URI);


    static void sax_warning     (void *ctx, const char *msg, ...);
    static void sax_error       (void *ctx, const char *msg, ...);
    static void sax_fatal_error (void *ctx, const char *msg, ...);

    // Callbacks for reading XML data
    void read_mesh        (const xmlChar* name, const xmlChar** attrs, uint num_attributes);
    void read_vertices    (const xmlChar* name, const xmlChar** attrs, uint num_attributes);
    void read_vertex      (const xmlChar* name, const xmlChar** attrs, uint num_attributes);
    void read_cells       (const xmlChar* name, const xmlChar** attrs, uint num_attributes);
    void read_interval    (const xmlChar* name, const xmlChar** attrs, uint num_attributes);
    void read_triangle    (const xmlChar* name, const xmlChar** attrs, uint num_attributes);
    void read_tetrahedron (const xmlChar* name, const xmlChar** attrs, uint num_attributes);
    /*
    void read_mesh_function(const xmlChar* name, const xmlChar** attrs);
    void read_mesh_data    (const xmlChar* name, const xmlChar** attrs);
    void read_data_entry   (const xmlChar* name, const xmlChar** attrs);
    */

    // Number of local vertices
    uint num_local_vertices() const;

    // Number of local cells
    uint num_local_cells() const;

    // Geometrical mesh dimesion
    uint gdim;

    // Topological mesh dimesion
    uint tdim;

    // Range for vertices
    std::pair<uint, uint> vertex_range;

    // Range for cells
    std::pair<uint, uint> cell_range;

    // State of parser
    ParserState state;

    LocalMeshData& mesh_data;

    const std::string filename;

  };

}
#endif

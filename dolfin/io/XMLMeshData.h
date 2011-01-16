// Copyright (C) 2009 Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-09
// Last changed: 2009-03-11

#ifndef __XMLMESHDATA_H
#define __XMLMESHDATA_H

#include <map>
#include <vector>
#include "XMLHandler.h"

namespace dolfin
{

  class MeshData;
  class XMLMeshFunction;
  class XMLArray;
  class XMLMap;

  class XMLMeshData: public XMLHandler
  {
  public:

    /// Constructor
    XMLMeshData(MeshData& data, XMLFile& parser, bool inside=false);

    /// Destructor
    ~XMLMeshData();

    void start_element (const xmlChar* name, const xmlChar** attrs);
    void end_element   (const xmlChar* name);

    static void write(const MeshData& data, std::ostream& outfile, uint indentation_level=0);

  private:
    enum parser_state {OUTSIDE, INSIDE_DATA, INSIDE_DATA_ENTRY, DONE};
    enum data_entry_type {ARRAY, MAP, ARRAY_MAP, MESH_FUNCTION, UNSET};

    void read_data_entry(const xmlChar* name, const xmlChar** attrs);
    void read_array(const xmlChar* name, const xmlChar** attrs);
    void read_map(const xmlChar* name, const xmlChar** attrs);
    void read_mesh_function(const xmlChar* name, const xmlChar** attrs);

    MeshData& data;
    parser_state state;
    data_entry_type type;
    std::string entity_name;

    XMLArray* xml_array;
    XMLMap* xml_map;
    XMLMeshFunction* xml_mesh_function;
    std::map<uint, int>*    im;
    std::map<uint, uint>*   um;
    std::map<uint, double>* dm;
    std::map<uint, std::vector<int> >*    iam;
    std::map<uint, std::vector<uint> >*   uam;
    std::map<uint, std::vector<double> >* dam;
  };

}
#endif

// Copyright (C) 2009 Ola Skavhaug
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
  class OldXMLFile;

  class XMLMeshData: public XMLHandler
  {
  public:

    /// Constructor
    XMLMeshData(MeshData& data, OldXMLFile& parser, bool inside=false);

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

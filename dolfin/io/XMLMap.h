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
// Modified by Garth N. Wells, 2009.
//
// First added:  2009-03-02
// Last changed: 2009-06-16

#ifndef __XMLMAP_H
#define __XMLMAP_H

#include <map>
#include <vector>
#include "XMLHandler.h"

namespace dolfin
{

  class OldXMLFile;
  class XMLArray;

  class XMLMap : public XMLHandler
  {
  public:

    XMLMap(std::map<uint, int>& im, OldXMLFile& parser);
    XMLMap(std::map<uint, uint>& um, OldXMLFile& parser);
    XMLMap(std::map<uint, double>& dm, OldXMLFile& parser);
    XMLMap(std::map<uint, std::vector<int> >& iam, OldXMLFile& parser);
    XMLMap(std::map<uint, std::vector<uint> >& uam, OldXMLFile& parser);
    XMLMap(std::map<uint, std::vector<double> >& dam, OldXMLFile& parser);

    // Callbacks
    void start_element (const xmlChar *name, const xmlChar **attrs);
    void end_element   (const xmlChar *name);

    /// Write to file
    static void write(const std::map<uint, int>& map, std::ostream& outfile,
                      uint indentation_level=0);
    static void write(const std::map<uint, uint>& map, std::ostream& outfile,
                      uint indentation_level=0);
    static void write(const std::map<uint, double>& map, std::ostream& outfile,
                      uint indentation_level=0);
    static void write(const std::map<uint, std::vector<int> >& map,
                      std::ostream& outfile, uint indentation_level=0);
    static void write(const std::map<uint, std::vector<uint> >& map,
                      std::ostream& outfile, uint indentation_level=0);
    static void write(const std::map<uint, std::vector<double> >& map,
                      std::ostream& outfile, uint indentation_level=0);

  private:

    enum parser_state { OUTSIDE_MAP, INSIDE_MAP, INSIDE_MAP_ENTRY, MAP_DONE };
    enum map_type { INT, UINT, DOUBLE, INT_ARRAY, UINT_ARRAY, DOUBLE_ARRAY, UNSET };

    void finalize_map_entry();
    void start_map(const xmlChar *name, const xmlChar **attrs);
    void read_map_entry(const xmlChar *name, const xmlChar **attrs);
    void read_int(const xmlChar *name, const xmlChar **attrs);
    void read_uint(const xmlChar *name, const xmlChar **attrs);
    void read_double(const xmlChar *name, const xmlChar **attrs);
    void read_array(const xmlChar *name, const xmlChar **attrs);
    void read_int_array(const xmlChar *name, const xmlChar **attrs, uint size);
    void read_uint_array(const xmlChar *name, const xmlChar **attrs, uint size);
    void read_double_array(const xmlChar *name, const xmlChar **attrs, uint size);

    // Basic maps
    std::map<uint, int>*  im;
    std::map<uint, uint>* um;
    std::map<uint, double>* dm;

    // Array maps
    std::map<uint, std::vector<int> >* iam;
    std::map<uint, std::vector<uint> >* uam;
    std::map<uint, std::vector<double> >* dam;

    std::vector<int>* ix;
    std::vector<uint>* ux;
    std::vector<double>* dx;
    XMLArray* xml_array;

    parser_state state;
    map_type mtype;

    // The current key
    uint current_key;

  };

}

#endif

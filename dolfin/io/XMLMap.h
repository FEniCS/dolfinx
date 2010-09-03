// Copyright (C) 2009 Ola Skavhaug
// Licensed under the GNU LGPL Version 2.1.
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

  class XMLFile;
  class XMLArray;

  class XMLMap : public XMLHandler
  {
  public:

    XMLMap(std::map<uint, int>& im, XMLFile& parser);
    XMLMap(std::map<uint, uint>& um, XMLFile& parser);
    XMLMap(std::map<uint, double>& dm, XMLFile& parser);
    XMLMap(std::map<uint, std::vector<int> >& iam, XMLFile& parser);
    XMLMap(std::map<uint, std::vector<uint> >& uam, XMLFile& parser);
    XMLMap(std::map<uint, std::vector<double> >& dam, XMLFile& parser);

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

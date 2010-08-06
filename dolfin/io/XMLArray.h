// Copyright (C) 2009 Ola Skavhaug
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-02
// Last changed: 2009-03-11

#ifndef __XMLARRAY_H
#define __XMLARRAY_H

#include <vector>
#include "XMLHandler.h"

namespace dolfin
{

  template<class T> class Array;

  class XMLArray : public XMLHandler
  {
  public:

    XMLArray(std::vector<int>& ix, XMLFile& parser);
    XMLArray(std::vector<int>& ix, XMLFile& parser, uint size);

    XMLArray(std::vector<uint>& ux, XMLFile& parser);
    XMLArray(std::vector<uint>& ux, XMLFile& parser, uint size);

    XMLArray(std::vector<double>& dx, XMLFile& parser);
    XMLArray(std::vector<double>& dx, XMLFile& parser, uint size);

    void start_element (const xmlChar *name, const xmlChar **attrs);
    void end_element   (const xmlChar *name);


    /// Write to file
    static void write(const std::vector<int>& x, uint offset,
                      std::ostream& outfile, uint indentation_level=0);
    static void write(const std::vector<uint>& x, uint offset,
                      std::ostream& outfile, uint indentation_level=0);
    static void write(const std::vector<double>& x, uint ofsfet,
                      std::ostream& outfile, uint indentation_level=0);
    static void write(const Array<double>& x, uint ofsfet,
                      std::ostream& outfile, uint indentation_level=0);

    void read_array_tag(const xmlChar *name, const xmlChar **attrs);

  private:

    enum parser_state { OUTSIDE_ARRAY, INSIDE_ARRAY, ARRAY_DONE };
    enum array_type { INT, UINT, DOUBLE, UNSET };

    void read_entry  (const xmlChar *name, const xmlChar **attrs);

    // Integer data
    std::vector<int>*  ix;

    // Unsigned integer data
    std::vector<uint>* ux;

    // Double data
    std::vector<double>* dx;

    parser_state state;
    array_type atype;

    uint size;

  };

}

#endif

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
// First added:  2009-03-02
// Last changed: 2009-03-11

#ifndef __XMLARRAY_H
#define __XMLARRAY_H

#include <utility>
#include <vector>
#include "XMLHandler.h"

namespace dolfin
{

  template<class T> class Array;

  class XMLArray : public XMLHandler
  {
  public:

    XMLArray(std::vector<int>& ix, XMLFile& parser, bool distributed = false);
    XMLArray(std::vector<int>& ix, XMLFile& parser, uint size, bool distributed = false);

    XMLArray(std::vector<uint>& ux, XMLFile& parser, bool distributed = false);
    XMLArray(std::vector<uint>& ux, XMLFile& parser, uint size, bool distributed = false);

    XMLArray(std::vector<double>& dx, XMLFile& parser, bool distributed = false);
    XMLArray(std::vector<double>& dx, XMLFile& parser, uint size, bool distributed = false);

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

    friend class XMLVector;

    enum parser_state { OUTSIDE_ARRAY, INSIDE_ARRAY, ARRAY_DONE };
    enum array_type { INT, UINT, DOUBLE, UNSET };

    void read_entry(const xmlChar *name, const xmlChar **attrs);

    // Element indices
    std::vector<uint> element_index;

    // Integer data
    std::vector<int>* ix;

    // Unsigned integer data
    std::vector<uint>* ux;

    // Double data
    std::vector<double>* dx;

    parser_state state;
    array_type atype;

    // Range for indices
    std::pair<uint, uint> range;

    uint size;

    bool distributed;

  };

}

#endif

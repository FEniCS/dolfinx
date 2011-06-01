// Copyright (C) 2009 Anders Logg and Ola Skavhaug
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2009-03-06
// Last changed: 2011-06-01

#ifndef __XMLFUNCTIONPLOTDATA_H
#define __XMLFUNCTIONPLOTDATA_H

#include "XMLHandler.h"

namespace dolfin
{

  class XMLVector;
  class FunctionPlotData;
  class XMLMesh;

  class XMLFunctionPlotData : public XMLHandler
  {
  public:

    XMLFunctionPlotData(FunctionPlotData& data, XMLFile& parser);
    ~XMLFunctionPlotData();

    void start_element (const xmlChar* name, const xmlChar** attrs);
    void end_element   (const xmlChar* name);

    static void write(const FunctionPlotData& data,
                      std::ostream& outfile,
                      uint indentation_level=0);

    void read_data_tag(const xmlChar* name, const xmlChar** attrs);

  private:

    enum parser_state {OUTSIDE, INSIDE, DONE};

    void read_mesh      (const xmlChar* name, const xmlChar** attrs);
    void read_vector    (const xmlChar* name, const xmlChar** attrs);

    FunctionPlotData& data;
    parser_state state;

    XMLMesh* xml_mesh;
    XMLVector* xml_vector;

  };

}

#endif

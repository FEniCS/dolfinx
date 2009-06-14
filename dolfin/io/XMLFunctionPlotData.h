// Copyright (C) 2009 Anders Logg and Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-06
// Last changed:  2009-03-17

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

    static void write(const FunctionPlotData& data, std::ostream& outfile, uint indentation_level=0);

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

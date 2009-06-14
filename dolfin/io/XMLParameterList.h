// Copyright (C) 2009 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-16
// Last changed: 2009-03-16

#ifndef __XMLPARAMETERLIST_H
#define __XMLPARAMETERLIST_H

#include "XMLHandler.h"

namespace dolfin
{

  class ParameterList;
  class XMLFile;

  class XMLParameterList : public XMLHandler
  {
  public:

    XMLParameterList(ParameterList& parameters, XMLFile& parser);

    void start_element(const xmlChar *name, const xmlChar **attrs);
    void end_element  (const xmlChar *name);

    static void write(const ParameterList& parameters, std::ostream& outgile, uint indentation_level=0);

  private:

    enum parser_state { OUTSIDE, INSIDE_PARAMETERS, DONE };

    void read_parameter(const xmlChar *name, const xmlChar **attrs);

    ParameterList& parameters;
    parser_state state;

  };

}

#endif

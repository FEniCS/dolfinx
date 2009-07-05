// Copyright (C) 2009 Ola Skavhaug and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-16
// Last changed: 2009-07-03

#ifndef __XML_PARAMETERS_H
#define __XML_PARAMETERS_H

#include "XMLHandler.h"

namespace dolfin
{

  class NewParameters;
  class XMLFile;

  // FIXME: Need to handle nested parameters and ranges in XML format

  class XMLParameters : public XMLHandler
  {
  public:

    XMLParameters(NewParameters& parameters, XMLFile& parser);

    void start_element(const xmlChar *name, const xmlChar **attrs);
    void end_element  (const xmlChar *name);

    static void write(const NewParameters& parameters, std::ostream& outgile, uint indentation_level=0);

  private:

    enum parser_state { OUTSIDE, INSIDE_PARAMETERS, DONE };

    void read_parameter(const xmlChar *name, const xmlChar **attrs);

    NewParameters& parameters;
    parser_state state;

  };

}

#endif

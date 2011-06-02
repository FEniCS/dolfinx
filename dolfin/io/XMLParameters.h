// Copyright (C) 2011 Ola Skavhaug and Anders Logg
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
// First added:  2009-03-16
// Last changed: 2011-03-28

#ifndef __XML_PARAMETERS_H
#define __XML_PARAMETERS_H

#include <boost/scoped_ptr.hpp>
#include "XMLHandler.h"

namespace dolfin
{

  class Parameters;
  class XMLFile;

  // FIXME: Need to handle nested parameters and ranges in XML format

  class XMLParameters : public XMLHandler
  {
  public:

    XMLParameters(Parameters& parameters, XMLFile& parser, bool inside=false);

    void start_element(const xmlChar *name, const xmlChar **attrs);
    void end_element  (const xmlChar *name);

    static void write(const Parameters& parameters, std::ostream& outgile, uint indentation_level=0);

  private:

    enum parser_state {OUTSIDE, INSIDE_PARAMETERS, DONE};

    void read_parameters(const xmlChar *name, const xmlChar **attrs);
    void read_nested_parameters(const xmlChar *name, const xmlChar **attrs);
    void read_parameter(const xmlChar *name, const xmlChar **attrs);

    Parameters& parameters;
    parser_state state;

    boost::scoped_ptr<XMLParameters> xml_nested_parameters;

  };

}

#endif

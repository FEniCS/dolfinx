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
// Modified by Anders Logg, 2011.
//
// First added:  2009-03-03
// Last changed: 2011-03-31

#ifndef __SAXHANDLER_H
#define __SAXHANDLER_H

#include <string>
#include <boost/lexical_cast.hpp>
#include <libxml/parser.h>

namespace dolfin
{

  class GenericVector;
  class Mesh;
  class Parameters;

  class SAX2AttributeParser
  {
  public:

    template<class T>
    static T parse(const xmlChar* name, const xmlChar** attrs,
			                 const char* attribute, uint num_attributes)
    {
      // Check that we got the data
      if (!attrs)
        error("Missing attribute \"%s\" for <%s> in XML file.  No attribute list given.",
                      attribute, name);

      // Parse data
      for (uint i = 0; i < num_attributes; i++)
      {
        // Check for attribute
        if (xmlStrcasecmp(attrs[5*i], (xmlChar *) attribute) == 0)
        {
          if (!attrs[5*i + 3])
          {
            error("Value for attribute \"%s\" of <%s> missing in XML file.",
		          attribute, name);
          }
          std::string value(attrs[5*i + 3], attrs[5*i + 4]);
          return boost::lexical_cast<T, std::string>(value);
        }
      }

      // Didn't get the value
      error("Missing attribute value for \"%s\" for <%s> in XML file.",
             attribute, name);

      return boost::lexical_cast<T, std::string>("0");
    }

  };

}
#endif

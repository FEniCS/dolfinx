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

#include <string>

namespace dolfin
{

  class Parameters;

  // FIXME: Need to handle nested parameters and ranges in XML format

  class XMLParameters
  {
  public:

    /// Read parameters from XML file
    static void read(Parameters& parameters, const pugi::xml_node xml_dolfin);

    /// Write parameters to XML file
    static void write(const Parameters& parameters, std::ostream& outfile,
                      unsigned int indentation_level=0);

  private:

    template<class T>
    static void add_parameter(Parameters& p, const std::string& key, T value);

  };

}

#endif

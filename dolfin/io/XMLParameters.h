// Copyright (C) 2011 Ola Skavhaug, Anders Logg and Garth N. Wells
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

#include <ostream>
#include <string>

namespace pugi
{
  class xml_node;
}

namespace dolfin
{

  class Parameters;

  // FIXME: Need to test nested parameters and ranges in XML format

  /// I/O of Parameters in XML format

  class XMLParameters
  {
  public:

    /// Read parameters from XML file
    static void read(Parameters& parameters, const pugi::xml_node xml_dolfin);

    /// Write the XML file
    static void write(const Parameters& parameters, pugi::xml_node xml_node);

  private:

    static void read_parameter_nest(Parameters& p, const pugi::xml_node xml_node);

    template<typename T>
    static void add_parameter(Parameters& p, const std::string& key, T value);

  };

}

#endif

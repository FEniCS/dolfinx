// Copyright (C) 2011 Garth N. Wells
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
// First added:  2003-07-15
// Last changed: 2006-05-23

#ifndef __XMLMESHFUNCTIONPLOTDATA_H
#define __XMLMESHFUNCTIONPLOTDATA_H

#include <ostream>

namespace pugi
{
  class xml_node;
}

namespace dolfin
{

  class FunctionPlotData;

  class XMLFunctionPlotData
  {
  public:

    // Read XML vector
    static void read(FunctionPlotData& plot_data,
                     const pugi::xml_node xml_dolfin);

    /// Write the XML file
    static void write(const FunctionPlotData& data, pugi::xml_node xml_node);

  };

}

#endif

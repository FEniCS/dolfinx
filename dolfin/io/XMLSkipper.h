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
// First added:  2009-08-06
// Last changed: 2009-08-06

#ifndef __XMLSKIPPER_H
#define __XMLSKIPPER_H

#include "XMLHandler.h"

namespace dolfin
{

  class XMLSkipper: public XMLHandler
  {
    public:
      XMLSkipper(std::string name, OldXMLFile& parser);
      void start_element(const xmlChar* name, const xmlChar** attrs) {/* Do nothing */}
      void end_element(const xmlChar* name);

    private:
      std::string name;
  };

}

#endif

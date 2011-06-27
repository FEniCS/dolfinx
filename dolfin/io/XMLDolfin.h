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
// First added:  2009-03-27
// Last changed: 2009-04-01

#ifndef __XMLDOLFIN_H
#define __XMLDOLFIN_H

#include "XMLHandler.h"

namespace dolfin
{

  class OldXMLFile;

  /// This class is responsible for the outer <dolfin> xml namespace,
  /// and dispatches the correct reader in start_element. It is needed
  /// when we store more than one data type pr file.

  class XMLDolfin : public XMLHandler
  {
  public:

    XMLDolfin(XMLHandler& dispatch, OldXMLFile& parser);

    void start_element (const xmlChar* name, const xmlChar** attrs);
    void end_element   (const xmlChar* name);

  private:

    enum parser_state { OUTSIDE_DOLFIN, INSIDE_DOLFIN};

    void read_entry  (const xmlChar* name, const xmlChar** attrs);

    parser_state state;
    XMLHandler& dispatch;
  };

}

#endif

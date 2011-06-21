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
// First added:  2009-03-02
// Last changed: 2009-03-17

#include <dolfin/log/dolfin_log.h>
#include "OldXMLFile.h"
#include "XMLIndent.h"
#include "XMLDolfin.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLDolfin::XMLDolfin(XMLHandler& dispatch, OldXMLFile& parser)
  : XMLHandler(parser), state(OUTSIDE_DOLFIN), dispatch(dispatch)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void XMLDolfin::start_element(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state )
  {
  case OUTSIDE_DOLFIN:

    if ( xmlStrcasecmp(name, (xmlChar *) "dolfin") == 0 )
    {
      state = INSIDE_DOLFIN;
      dispatch.handle();
    }
    break;

  default:
    break;
  }
}
//-----------------------------------------------------------------------------
void XMLDolfin::end_element(const xmlChar *name)
{
  switch ( state )
  {
  case INSIDE_DOLFIN:

    if ( xmlStrcasecmp(name, (xmlChar *) "dolfin") == 0 )
    {
      state = OUTSIDE_DOLFIN;
      release();
    }
    break;

  default:
    break;
  }
}
//-----------------------------------------------------------------------------
void XMLDolfin::write_start(std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent() << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl << std::endl;
  outfile << indent() << "<dolfin xmlns:dolfin=\"http://www.fenicsproject.org\">" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLDolfin::write_end(std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent() << "</dolfin>" << std::endl;
}
//-----------------------------------------------------------------------------

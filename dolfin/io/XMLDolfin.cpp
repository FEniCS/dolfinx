// Copyright (C) 2009 Ola Skavhaug
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-02
// Last changed: 2009-03-17

#include <dolfin/log/dolfin_log.h>
#include "XMLFile.h"
#include "XMLIndent.h"
#include "XMLDolfin.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLDolfin::XMLDolfin(XMLHandler& dispatch, XMLFile& parser)
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
  outfile << indent() << "<dolfin xmlns:dolfin=\"http://www.fenics.org/dolfin/\">" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLDolfin::write_end(std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);
  outfile << indent() << "</dolfin>" << std::endl;
}
//-----------------------------------------------------------------------------

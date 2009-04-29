// Copyright (C) 2009 Ola Skavhaug
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-27
// Last changed: 2009-04-01

#ifndef __XMLDOLFIN_H
#define __XMLDOLFIN_H

#include "XMLHandler.h"

namespace dolfin
{

  /// This class is responsible for the outer <dolfin> xml namespace,
  /// and dispatches the correct reader in start_element. It is needed
  /// when we store more than one data type pr file.

  class XMLDolfin : public XMLHandler
  {
  public:

    XMLDolfin(XMLHandler& dispatch, NewXMLFile& parser);

    void start_element (const xmlChar *name, const xmlChar **attrs);
    void end_element   (const xmlChar *name);


    /// Write to file
    static void write_start(std::ostream& outfile, uint indentation_level=0);
    static void write_end(std::ostream& outfile, uint indentation_level=0);

  private:

    enum parser_state { OUTSIDE_DOLFIN, INSIDE_DOLFIN};

    void read_entry  (const xmlChar *name, const xmlChar **attrs);

    parser_state state;
    XMLHandler& dispatch;
  };

}

#endif

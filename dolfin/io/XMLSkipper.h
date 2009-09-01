// Copyright (C) 2009 Ola Skavhaug
// Licensed under the GNU LGPL Version 2.1.
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
      XMLSkipper(std::string name, XMLFile& parser);
      void start_element(const xmlChar* name, const xmlChar** attrs) {/* Do nothing */}
      void end_element(const xmlChar* name);

    private:
      std::string name;
  };

}

#endif

// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-07-15
// Last changed: 2007-05-15

#ifndef __XML_VECTOR_H
#define __XML_VECTOR_H

#include <dolfin/la/Vector.h>
#include "XMLArray.h"
#include "XMLHandler.h"

namespace dolfin
{

  class NewXMLVector : public XMLHandler
  {
  public:

    NewXMLVector(GenericVector& vector, NewXMLFile& parser, bool inside=false);
    
    void start_element (const xmlChar *name, const xmlChar **attrs);
    void end_element   (const xmlChar *name);
    
  private:
    
    enum parser_state { OUTSIDE, INSIDE_VECTOR, DONE };
    
    void start_vector(const xmlChar *name, const xmlChar **attrs);
    void end_vector  ();
    
    GenericVector& x;
    parser_state state;

    std::vector<double>* values;
    XMLArray* xml_array;

  };
  
}

#endif

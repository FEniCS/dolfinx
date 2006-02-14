// Copyright (C) 2003-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-07-15
// Last changed: 2006-02-13

#ifndef __XML_VECTOR_H
#define __XML_VECTOR_H

#include "XMLObject.h"

namespace dolfin
{

  class Vector;
  
  class XMLVector : public XMLObject
  {
  public:

    XMLVector(Vector& vector);
    
    void startElement (const xmlChar *name, const xmlChar **attrs);
    void endElement   (const xmlChar *name);
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_VECTOR, DONE };
    
    void readVector  (const xmlChar *name, const xmlChar **attrs);
    void readEntry   (const xmlChar *name, const xmlChar **attrs);
    
    Vector& x;
    ParserState state;
    
  };
  
}

#endif

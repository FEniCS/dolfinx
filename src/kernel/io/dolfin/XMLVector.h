// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-07-15
// Last changed: 2007-05-15

#ifndef __XML_VECTOR_H
#define __XML_VECTOR_H

#include <dolfin/Vector.h>
#include <dolfin/XMLObject.h>

namespace dolfin
{
  
  class XMLVector : public XMLObject
  {
  public:

    XMLVector(Vector& vector);
    
    void startElement (const xmlChar *name, const xmlChar **attrs);
    void endElement   (const xmlChar *name);
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_VECTOR, DONE };
    
    void startVector(const xmlChar *name, const xmlChar **attrs);
    void endVector  ();
    void readEntry  (const xmlChar *name, const xmlChar **attrs);
    
    Vector& x;
    ParserState state;

    real* values;
    uint size;

  };
  
}

#endif

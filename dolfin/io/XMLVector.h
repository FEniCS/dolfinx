// Copyright (C) 2003-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2003-07-15
// Last changed: 2007-05-15

#ifndef __XML_VECTOR_H
#define __XML_VECTOR_H

#include <dolfin/la/Vector.h>
#include "XMLObject.h"

namespace dolfin
{

  class XMLMesh;

  class XMLVector : public XMLObject
  {
  public:

    XMLVector(GenericVector& vector);
    
    void startElement (const xmlChar *name, const xmlChar **attrs);
    void endElement   (const xmlChar *name);
    
    /// Friends
    friend class XMLMesh;
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_VECTOR, DONE };
    
    void startVector(const xmlChar *name, const xmlChar **attrs);
    void endVector  ();
    void readEntry  (const xmlChar *name, const xmlChar **attrs);
    
    GenericVector& x;
    ParserState state;

    double* values;
    uint size;

  };
  
}

#endif

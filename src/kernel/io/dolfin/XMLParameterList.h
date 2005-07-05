// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-03-31
// Last changed: 2005

#ifndef __XML_PARAMETER_LIST_H
#define __XML_PARAMETER_LIST_H

#include <dolfin/XMLObject.h>

namespace dolfin {

  class ParameterList;
  
  class XMLParameterList : public XMLObject {
  public:

    XMLParameterList(ParameterList& parameters);
    
    void startElement (const xmlChar *name, const xmlChar **attrs);
    void endElement   (const xmlChar *name);
    
    void reading(std::string filename);
    void done();
    
  private:
    
    enum ParserState { OUTSIDE, INSIDE_PARAMETERS, DONE };
    
    void readParameter(const xmlChar *name, const xmlChar **attrs);
    
    ParameterList& parameters;
    ParserState state;
    
  };
  
}

#endif

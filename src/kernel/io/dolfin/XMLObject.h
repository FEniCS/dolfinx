// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003-07-15
// Last changed: 2005-10-03

#ifndef __XML_OBJECT_H
#define __XML_OBJECT_H

#include <libxml/parser.h>
#include <string>

#include <dolfin/constants.h>

namespace dolfin
{

  class XMLObject
  {
  public:
    
    XMLObject();
    
    virtual void startElement (const xmlChar* name, const xmlChar** attrs) = 0;
    virtual void endElement   (const xmlChar* name) = 0;
    
    // Write message before and after reading file
    virtual void reading(std::string filename) {};
    virtual void done() {};
    virtual ~XMLObject() {};
    
    bool dataOK();
    
  protected:
    
    void parseIntegerRequired (const xmlChar* name, const xmlChar** attrs, 
			       const char *attribute, int& value);

    void parseIntegerOptional (const xmlChar* name, const xmlChar** attrs,
			       const char* attribute, int& value);

    void parseRealRequired    (const xmlChar* name, const xmlChar** attrs,
			       const char* attribute, real& value);

    void parseRealOptional    (const xmlChar* name, const xmlChar** attrs,
			       const char* attribute, real& value);

    void parseStringRequired  (const xmlChar* name, const xmlChar** attrs,
			       const char* attribute, std::string& value);

    void parseStringOptional  (const xmlChar* name, const xmlChar** attrs,
			       const char* attribute, std::string& value);
    
    bool ok;
    
  };
  
}

#endif

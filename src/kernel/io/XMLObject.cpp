// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include "XMLObject.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLObject::XMLObject()
{
  ok = false;
}
//-----------------------------------------------------------------------------
bool XMLObject::dataOK()
{
  return ok;
}
//-----------------------------------------------------------------------------
void XMLObject::parseIntegerRequired(const xmlChar *name, const xmlChar **attrs,
												 const char *attribute, int *value)
{
  // Check that we got the data
  if ( !attrs )
	 dolfin_error1("Missing attributes for <%s> in XML file.", name);
  
  // Parse data
  for (int i = 0; attrs[i]; i++) {
	 
	 // Check for attribute
	 if ( xmlStrcasecmp(attrs[i], (xmlChar *) attribute) == 0 ){

		if ( !attrs[i+1] )
		  dolfin_error2("Value for attribute \"%s\" in <%s> is missing in XML file.", attribute, name);
		
		*value = atoi( (char *) attrs[i+1] );
		return;
	 }
	 
  }
  
  // Didn't get the value
  dolfin_error2("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);
}
//-----------------------------------------------------------------------------
void XMLObject::parseIntegerOptional(const xmlChar *name, const xmlChar **attrs,
											  const char *attribute, int *value)
{
  // Check if we got the data
  if ( !attrs )
	 return;
  
  // Parse data
  for (int i = 0; attrs[i]; i++) {
	 
	 // Check for attribute
	 if ( xmlStrcasecmp(attrs[i], (xmlChar *) attribute) == 0 ){
		if ( !attrs[i+1] )
		  dolfin_error2("Value for attribute \"%s\" in <%s> is missing in XML file.", attribute, name);
		*value = atoi( (char *) attrs[i+1] );
		return;
	 }
	 
  }
  
}
//-----------------------------------------------------------------------------
void XMLObject::parseRealRequired(const xmlChar *name, const xmlChar **attrs,
										  const char *attribute, real *value)
{
  // Check that we got the data
  if ( !attrs )
	 dolfin_error1("Missing attributes for <%s> in XML file.", name);
  
  // Parse data
  for (int i = 0; attrs[i]; i++){
	 
	 // Check for attribute
	 if ( xmlStrcasecmp(attrs[i],(xmlChar *) attribute) == 0 ){
		if ( !attrs[i+1] )
		  dolfin_error2("Value for attribute \"%s\" in <%s> is missing in XML file.", attribute, name);
		*value = (real) atof( (char *) attrs[i+1] );
		return;
	 }
	 
  }
  
  dolfin_error2("Missing attribute \"%s\" for <%s> in XML file.", attribute, name);
}
//-----------------------------------------------------------------------------
void XMLObject::parseRealOptional(const xmlChar *name, const xmlChar **attrs,
										  const char *attribute, real *value)
{
  // Check if we got the data
  if ( !attrs )
	 return;
  
  // Parse data
  for (int i = 0; attrs[i]; i++) {
	 
	 // Check for attribute
	 if ( xmlStrcasecmp(attrs[i],(xmlChar *) attribute) == 0 ){
		if ( !attrs[i+1] )
		  dolfin_error2("Value for attribute \"%s\" in <%s> is missing in XML file.", attribute, name);
		*value = (real) atof( (char *) attrs[i+1] );
		return;
	 }
	 
  }
  
}
//-----------------------------------------------------------------------------

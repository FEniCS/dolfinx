//FIXME: Temporary until we get the logsystem working
#include <iostream>

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
  if ( !attrs ){
  	 // FIXME: Temporary until we get the logsystem working
	 std::cout << "Missing attributes for <" << name << "> in XML file." << std::endl;
	 exit(1);
  }
  
  // Parse data
  for (int i=0;attrs[i];i++){
	 
	 // Check for attribute
	 if ( xmlStrcasecmp(attrs[i],(xmlChar *) attribute) == 0 ){
		if ( !attrs[i+1] ){
		  // FIXME: Temporary until we get the logsystem working
		  std::cout << "Value for attribute \"" << attribute << "\" in <"
						<< name << "> is missing in XML file ." << std::endl;
		  exit(1);
		}
		*value = atoi( (char *) attrs[i+1] );
		return;
	 }
	 
  }
  
  // Didn't get the value
  // FIXME: Temporary until we get the logsystem working
  std::cout << "Missing attribute \"" << attribute << "\" for <"
				<< name << "> in XML file." << std::endl;
  exit(1);
}
//-----------------------------------------------------------------------------
void XMLObject::parseIntegerOptional(const xmlChar *name, const xmlChar **attrs,
											  const char *attribute, int *value)
{
  // Check if we got the data
  if ( !attrs )
	 return;
  
  // Parse data
  for (int i=0;attrs[i];i++){
	 
	 // Check for attribute
	 if ( xmlStrcasecmp(attrs[i],(xmlChar *) attribute) == 0 ){
		if ( !attrs[i+1] ){
		  // FIXME: Temporary until we get the logsystem working
		  std::cout << "Value for attribute \"" << attribute << "\" in <"
						<< name << "> is missing in XML file ." << std::endl;
		  exit(1);
		}
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
  if ( !attrs ){
  	 // FIXME: Temporary until we get the logsystem working
	 std::cout << "Missing attributes for <"
				  << name << "> in XML file." << std::endl;
	 exit(1);
  }
  
  // Parse data
  for (int i=0;attrs[i];i++){
	 
	 // Check for attribute
	 if ( xmlStrcasecmp(attrs[i],(xmlChar *) attribute) == 0 ){
		if ( !attrs[i+1] ){
		  // FIXME: Temporary until we get the logsystem working
		  std::cout << "Value for attribute \"" << attribute << "\" in <"
						<< name << "> is missing in XML file ." << std::endl;
		  exit(1);
		}
		*value = (real) atof( (char *) attrs[i+1] );
		return;
	 }
	 
  }
  
  // Didn't get the value
  // FIXME: Temporary until we get the logsystem working
  std::cout << "Missing attribute \"" << attribute << "\" for <"
				<< name << "> in XML file." << std::endl;
  exit(1);
}
//-----------------------------------------------------------------------------
void XMLObject::parseRealOptional(const xmlChar *name, const xmlChar **attrs,
										  const char *attribute, real *value)
{
  // Check if we got the data
  if ( !attrs )
	 return;
  
  // Parse data
  for (int i=0;attrs[i];i++){
	 
	 // Check for attribute
	 if ( xmlStrcasecmp(attrs[i],(xmlChar *) attribute) == 0 ){
		if ( !attrs[i+1] ){
		  // FIXME: Temporary until we get the logsystem working
		  std::cout << "Value for attribute \"" << attribute << "\" in <"
						<< name << "> is missing in XML file ." << std::endl;
		  exit(1);
		}
		*value = (real) atof( (char *) attrs[i+1] );
		return;
	 }
	 
  }
  
}
//-----------------------------------------------------------------------------

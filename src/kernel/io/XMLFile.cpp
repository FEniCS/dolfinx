#include <dolfin/Vector.h>
#include "XMLFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLFile::XMLFile(const std::string filename) : GenericFile(filename)
{
  // Reset data
  resetData();
  
  // Set up handlers for parser events
  sax.startDocument = sax_start_document;
  sax.endDocument   = sax_end_document;
  sax.startElement  = sax_start_element;
  sax.endElement    = sax_end_element;
  sax.warning       = sax_warning;
  sax.error         = sax_error;
  sax.fatalError    = sax_fatal_error;
}
//-----------------------------------------------------------------------------
XMLFile::~XMLFile()
{
  
  

  
}
//-----------------------------------------------------------------------------
void XMLFile::operator>> (Vector& vector)
{
  this->vector = &vector;
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator<< (const Vector& vector)
{
  



}
//-----------------------------------------------------------------------------
void XMLFile::VectorInit(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  int size = 0;

  // Parse values
  parseIntegerRequired(name, attrs, "size", &size);

  // Check values
  if ( size < 0 ){
	 // FIXME: Temporary until we get the logsystem working
	 cout << "Error reading XML data: size of vector must be positive." << endl;
	 exit(1);
  }

  // Initialise
  vector->resize(size);	 
}
//-----------------------------------------------------------------------------
void XMLFile::VectorValue(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  int row = 0;
  real value = 0.0;
  
  // Parse values
  parseIntegerRequired(name, attrs, "row", &row);
  parseRealRequired(name, attrs, "value", &value);   
  
  // Check values
  if ( row < 0 || row >= vector->size() ){
	 // FIXME: Temporary until we get the logsystem working
	 cout << "Error reading XML data: row index " << row
			<< " for vector out of range (0 - " << vector->size()
			<< ")" << endl;
	 exit(1);
  }
  
  // Set value
  (*vector)(row) = value;
}
//-----------------------------------------------------------------------------
void XMLFile::resetData()
{
  vector = 0;
}
//-----------------------------------------------------------------------------
void XMLFile::parseFile()
{
  xmlSAXUserParseFile(&sax, this, filename.c_str());
}
//-----------------------------------------------------------------------------
void XMLFile::parseIntegerRequired(const xmlChar *name, const xmlChar **attrs,
											  const char *attribute, int *value)
{
  // Check that we got the data
  if ( !attrs ){
  	 // FIXME: Temporary until we get the logsystem working
	 cout << "Missing attributes for <" << name << "> in XML file." << endl;
	 exit(1);
  }
  
  // Parse data
  for (int i=0;attrs[i];i++){
	 
	 // Check for attribute
	 if ( xmlStrcasecmp(attrs[i],(xmlChar *) attribute) == 0 ){
		if ( !attrs[i+1] ){
		  // FIXME: Temporary until we get the logsystem working
		  cout << "Value for attribute \"" << attribute << "\" in <" << name << "> is missing in XML file ." << endl;
		  exit(1);
		}
		*value = atoi( (char *) attrs[i+1] );
		return;
	 }
	 
  }
  
  // Didn't get the value
  // FIXME: Temporary until we get the logsystem working
  cout << "Missing attribute \"" << attribute << "\" for <" << name << "> in XML file." << endl;
  exit(1);
}
//-----------------------------------------------------------------------------
void XMLFile::parseIntegerOptional(const xmlChar *name, const xmlChar **attrs,
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
		  cout << "Value for attribute \"" << attribute << "\" in <" << name << "> is missing in XML file ." << endl;
		  exit(1);
		}
		*value = atoi( (char *) attrs[i+1] );
		return;
	 }
	 
  }
  
}
//-----------------------------------------------------------------------------
void XMLFile::parseRealRequired(const xmlChar *name, const xmlChar **attrs,
										  const char *attribute, real *value)
{
  // Check that we got the data
  if ( !attrs ){
  	 // FIXME: Temporary until we get the logsystem working
	 cout << "Missing attributes for <" << name << "> in XML file." << endl;
	 exit(1);
  }
  
  // Parse data
  for (int i=0;attrs[i];i++){
	 
	 // Check for attribute
	 if ( xmlStrcasecmp(attrs[i],(xmlChar *) attribute) == 0 ){
		if ( !attrs[i+1] ){
		  // FIXME: Temporary until we get the logsystem working
		  cout << "Value for attribute \"" << attribute << "\" in <" << name << "> is missing in XML file ." << endl;
		  exit(1);
		}
		*value = (real) atof( (char *) attrs[i+1] );
		return;
	 }
	 
  }
  
  // Didn't get the value
  // FIXME: Temporary until we get the logsystem working
  cout << "Missing attribute \"" << attribute << "\" for <" << name << "> in XML file." << endl;
  exit(1);
}
//-----------------------------------------------------------------------------
void XMLFile::parseRealOptional(const xmlChar *name, const xmlChar **attrs,
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
		  cout << "Value for attribute \"" << attribute << "\" in <" << name << "> is missing in XML file ." << endl;
		  exit(1);
		}
		*value = (real) atof( (char *) attrs[i+1] );
		return;
	 }
	 
  }
  
}
//-----------------------------------------------------------------------------
namespace dolfin {
  
  //---------------------------------------------------------------------------
  void sax_start_document(void *ctx)
  {
  }
  //---------------------------------------------------------------------------
  void sax_end_document(void *ctx)
  {
  }
  //---------------------------------------------------------------------------
  void sax_start_element(void *ctx, const xmlChar *name, const xmlChar **attrs)
  {
	 if ( xmlStrcasecmp(name,(xmlChar *) "vector") == 0 )
		( (XMLFile *) ctx )->VectorInit(name,attrs);
	 else if ( xmlStrcasecmp(name,(xmlChar *) "element") == 0 )
		( (XMLFile *) ctx )->VectorValue(name,attrs);
  }
  //---------------------------------------------------------------------------
  void sax_end_element(void *ctx, const xmlChar *name)
  {
  }
  //---------------------------------------------------------------------------
  static void sax_warning(void *ctx, const char *msg, ...)
  {
	 va_list args;

	 va_start(args, msg);
	 // FIXME: Temporary until we get the logsystem working
	 printf("Warning from XML parser:\n");
	 vprintf(msg, args);
	 va_end(args);
  }
  //---------------------------------------------------------------------------
  static void sax_error(void *ctx, const char *msg, ...)
  {
	 va_list args;

	 va_start(args, msg);
	 // FIXME: Temporary until we get the logsystem working
	 printf("Error from XML parser:\n");
	 vprintf(msg, args);
	 va_end(args);
  }
  //---------------------------------------------------------------------------
  static void sax_fatal_error(void *ctx, const char *msg, ...)
  {
	 va_list args;
	 
	 va_start(args, msg);
	 // FIXME: Temporary until we get the logsystem working
	 printf("Fatal error from XML parser:\n");
	 vprintf(msg, args);
	 va_end(args);
  }
  //---------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------

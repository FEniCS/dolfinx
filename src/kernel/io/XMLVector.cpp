#include <dolfin/Vector.h>
#include "XMLVector.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLVector::XMLVector(Vector& vector) : XMLObject(), x(vector)
{
  state = OUTSIDE;
}
//-----------------------------------------------------------------------------
void XMLVector::startElement(const xmlChar *name, const xmlChar **attrs)
{
  switch ( state ) {
  case OUTSIDE:

	 if ( xmlStrcasecmp(name,(xmlChar *) "vector") == 0 ) {
		readVector(name,attrs);
		state = INSIDE_VECTOR;
	 }
	 
	 break;

  case INSIDE_VECTOR:
	 
	 if ( xmlStrcasecmp(name,(xmlChar *) "element") == 0 )
		readElement(name,attrs);
    	
	 break;

  default:
	 ;
  }
  
}
//-----------------------------------------------------------------------------
void XMLVector::endElement(const xmlChar *name)
{
  switch ( state ) {
  case INSIDE_VECTOR:
	 
	 if ( xmlStrcasecmp(name,(xmlChar *) "vector") == 0 ){
		ok = true;
		state = DONE;
	 }
	 
	 break;

  default:
	 ;
  }

}
//-----------------------------------------------------------------------------
void XMLVector::readVector(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  int size = 0;

  // Parse values
  parseIntegerRequired(name, attrs, "size", &size);

  // Check values
  if ( size < 0 ) {
	 // FIXME: Temporary until we get the logsystem working
	 cout << "Error reading XML data: size of vector must be positive." << endl;
	 exit(1);
  }

  // Initialise
  x.init(size);	 
}
//-----------------------------------------------------------------------------
void XMLVector::readElement(const xmlChar *name, const xmlChar **attrs)
{
  // Set default values
  int row = 0;
  real value = 0.0;
  
  // Parse values
  parseIntegerRequired(name, attrs, "row", &row);
  parseRealRequired(name, attrs, "value", &value);   
  
  // Check values
  if ( row < 0 || row >= x.size() ) {
	 // FIXME: Temporary until we get the logsystem working
	 cout << "Error reading XML data: row index " << row
			<< " for vector out of range (0 - " << x.size()
			<< ")" << endl;
	 exit(1);
  }
  
  // Set value
  x(row) = value;
}
//-----------------------------------------------------------------------------

#include <dolfin/Vector.h>
#include "XMLFile.h"
#include "XMLObject.h"
#include "XMLVector.h"
#include "XMLSparseMatrix.h"
#include "XMLGrid.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLFile::XMLFile(const std::string filename) : GenericFile(filename)
{
  xmlObject = 0;
}
//-----------------------------------------------------------------------------
XMLFile::~XMLFile()
{
  if ( xmlObject )
	 delete xmlObject;
}
//-----------------------------------------------------------------------------
void XMLFile::operator>>(Vector& vector)
{
  if ( xmlObject )
	 delete xmlObject;
  xmlObject = new XMLVector(&vector);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>>(SparseMatrix& sparseMatrix)
{
  if ( xmlObject )
	 delete xmlObject;
  xmlObject = new XMLSparseMatrix(&sparseMatrix);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>>(Grid& grid)
{
  if ( xmlObject )
	 delete xmlObject;
  xmlObject = new XMLGrid(&grid);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator<<(const Vector& vector)
{

}
//-----------------------------------------------------------------------------
void XMLFile::operator<<(const SparseMatrix& sparseMatrix)
{

}
//-----------------------------------------------------------------------------
void XMLFile::operator<<(const Grid& Grid)
{

}
//-----------------------------------------------------------------------------
void XMLFile::parseFile()
{
  // Write a message
  xmlObject->reading(filename);

  // Parse file using the SAX interface
  parseSAX();

  // Check that we got the data
  // FIXME: Temporary until we get the logsystem working  
  if ( !xmlObject->dataOK() ){
	 cout << "Unable to find data in XML file" << endl;
	 exit(1);
  }

  // Write a message
  xmlObject->done();
}
//-----------------------------------------------------------------------------
void XMLFile::parseSAX()
{
  // Set up the sax handler. Note that it is important that we initialise
  // all (24) fields, even the ones we don't use!
  xmlSAXHandler sax = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  
  // Set up handlers for parser events
  sax.startDocument = sax_start_document;
  sax.endDocument   = sax_end_document;
  sax.startElement  = sax_start_element;
  sax.endElement    = sax_end_element;
  sax.warning       = sax_warning;
  sax.error         = sax_error;
  sax.fatalError    = sax_fatal_error;
  
  // Parse file
  xmlSAXUserParseFile(&sax, (void *) xmlObject, filename.c_str());
}
//-----------------------------------------------------------------------------
// Callback functions for the SAX interface
//-----------------------------------------------------------------------------
void dolfin::sax_start_document(void *ctx)
{
  
}
//-----------------------------------------------------------------------------
void dolfin::sax_end_document(void *ctx)
{
  
}
//-----------------------------------------------------------------------------
void dolfin::sax_start_element(void *ctx,
										 const xmlChar *name, const xmlChar **attrs)
{
  ( (XMLObject *) ctx )->startElement(name, attrs);
}
//-----------------------------------------------------------------------------
void dolfin::sax_end_element(void *ctx, const xmlChar *name)
{
  ( (XMLObject *) ctx )->endElement(name);
}
//-----------------------------------------------------------------------------
void dolfin::sax_warning(void *ctx, const char *msg, ...)
{
  va_list args;
  
  va_start(args, msg);
  // FIXME: Temporary until we get the logsystem working
  printf("Warning from XML parser:\n");
  vprintf(msg, args);
  va_end(args);
}
//-----------------------------------------------------------------------------
void dolfin::sax_error(void *ctx, const char *msg, ...)
{
  va_list args;
  
  va_start(args, msg);
  // FIXME: Temporary until we get the logsystem working
  printf("Error from XML parser:\n");
  vprintf(msg, args);
  va_end(args);
}
//-----------------------------------------------------------------------------
void dolfin::sax_fatal_error(void *ctx, const char *msg, ...)
{
  va_list args;
  
  va_start(args, msg);
  // FIXME: Temporary until we get the logsystem working
  printf("Fatal error from XML parser:\n");
  vprintf(msg, args);
  va_end(args);
}
//-----------------------------------------------------------------------------

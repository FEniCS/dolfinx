#include <stdarg.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/Grid.h>
#include <dolfin/Function.h>

#include "XMLFile.h"
#include "XMLObject.h"
#include "XMLVector.h"
#include "XMLMatrix.h"
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
void XMLFile::operator>>(Vector& x)
{
  if ( xmlObject )
	 delete xmlObject;
  xmlObject = new XMLVector(x);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>>(Matrix& A)
{
  if ( xmlObject )
	 delete xmlObject;
  xmlObject = new XMLMatrix(A);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>>(Grid& grid)
{
  if ( xmlObject )
	 delete xmlObject;
  xmlObject = new XMLGrid(grid);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>>(Function& u)
{
  dolfin_warning("Cannot read functions from XML files.");
}
//-----------------------------------------------------------------------------
void XMLFile::operator<<(Vector& x)
{
  dolfin_warning("Cannot write vectors to XML files.");
}
//-----------------------------------------------------------------------------
void XMLFile::operator<<(Matrix& A)
{
  dolfin_warning("Cannot write matrices to XML files.");
}
//-----------------------------------------------------------------------------
void XMLFile::operator<<(Grid& Grid)
{
  dolfin_warning("Cannot write grids to XML files.");
}
//-----------------------------------------------------------------------------
void XMLFile::operator<<(Function& u)
{
  dolfin_warning("Cannot write functions to XML files.");
}
//-----------------------------------------------------------------------------
void XMLFile::parseFile()
{
  // Write a message
  xmlObject->reading(filename);

  // Parse file using the SAX interface
  parseSAX();

  // Check that we got the data
  if ( !xmlObject->dataOK() )
	 dolfin_error("Unable to find data in XML file.");

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

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
void XMLFile::operator>>(Vector& vector)
{
  xmlObject = new XMLVector(&vector);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>>(SparseMatrix& sparseMatrix)
{
  xmlObject = new XMLSparseMatrix(&sparseMatrix);
  parseFile();
}
//-----------------------------------------------------------------------------
void XMLFile::operator>>(Grid& grid)
{
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
  // Parse file
  xmlSAXUserParseFile(&sax, this, filename.c_str());

  // Check that we got the data
  // FIXME: Temporary until we get the logsystem working  
  if ( !xmlObject->dataOK() ){
	 cout << "Unable to find data in XML file" << endl;
	 exit(1);
  }
  
}
//-----------------------------------------------------------------------------

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
	 ( (XMLFile *) ctx )->xmlObject->startElement(name, attrs);
  }
  //---------------------------------------------------------------------------
  void sax_end_element(void *ctx, const xmlChar *name)
  {
	 ( (XMLFile *) ctx )->xmlObject->endElement(name);
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

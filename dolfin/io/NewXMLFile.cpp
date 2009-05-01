// Copyright (C) 2009 Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added: 2009-03-03
// Last changed: 2009-03-11

#include <dolfin/common/types.h>
#include <dolfin/common/constants.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/plot/FunctionPlotData.h>
#include "XMLArray.h"
#include "XMLMap.h"
#include "NewXMLFile.h"
#include "NewXMLMesh.h"
#include "NewXMLLocalMeshData.h"
#include "NewXMLGraph.h"
#include "NewXMLMatrix.h"
#include "XMLFunctionPlotData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NewXMLFile::NewXMLFile(const std::string filename, bool gzip)
  : GenericFile(filename), sax(0), outstream(0)
{
  // Set up the output stream (to file)
  outstream = new std::ofstream();

  // Set up the sax handler.
  sax = new xmlSAXHandler();

  // Set up handlers for parser events
  sax->startDocument = new_sax_start_document;
  sax->endDocument   = new_sax_end_document;
  sax->startElement  = new_sax_start_element;
  sax->endElement    = new_sax_end_element;
  sax->warning       = new_sax_warning;
  sax->error         = new_sax_error;
  sax->fatalError    = new_sax_fatal_error;
}
//-----------------------------------------------------------------------------
NewXMLFile::NewXMLFile(std::ostream& s)
  : GenericFile(""), sax(0), outstream(&s)
{
  // Set up the sax handler.
  sax = new xmlSAXHandler();

  // Set up handlers for parser events
  sax->startDocument = new_sax_start_document;
  sax->endDocument   = new_sax_end_document;
  sax->startElement  = new_sax_start_element;
  sax->endElement    = new_sax_end_element;
  sax->warning       = new_sax_warning;
  sax->error         = new_sax_error;
  sax->fatalError    = new_sax_fatal_error;
}
//-----------------------------------------------------------------------------
NewXMLFile::~NewXMLFile()
{
  delete sax;

  // Only delete outstream if it is a ofstream
  std::ofstream* outfile = dynamic_cast<std::ofstream*>(outstream);
  if ( outfile != 0)
  {
    outfile = 0;
    delete outstream;
  }
}
//-----------------------------------------------------------------------------
void NewXMLFile::validate(const std::string filename)
{
  xmlRelaxNGParserCtxtPtr parser;
  xmlRelaxNGValidCtxtPtr validator;
  xmlRelaxNGPtr schema;
  xmlDocPtr document;
  document = xmlParseFile(filename.c_str());
  int ret = 1;
  parser = xmlRelaxNGNewParserCtxt("http://fenics.org/pub/misc/dolfin.rng");
  xmlRelaxNGSetParserStructuredErrors(parser,
                                      (xmlStructuredErrorFunc)new_rng_parser_error,
                                      stderr);
  schema = xmlRelaxNGParse(parser);
  validator = xmlRelaxNGNewValidCtxt(schema);
  xmlRelaxNGSetValidStructuredErrors(validator,
                                     (xmlStructuredErrorFunc)new_rng_valid_error,
                                     stderr);
  ret = xmlRelaxNGValidateDoc(validator, document);
  if ( ret == 0 ) {
    message(0, "%s validates", filename.c_str());
  }
  else if ( ret < 0 ) {
    error("%s failed to load", filename.c_str());
  }
  else {
    error("%s fails to validate", filename.c_str());
  }
  xmlRelaxNGFreeValidCtxt(validator);
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator>>(std::vector<int>& x)
{
  message(1, "Reading array from file %s.", filename.c_str());
  XMLArray xml_array(x, *this);
  XMLDolfin xml_dolfin(xml_array, *this);
  xml_dolfin.handle();
  parse();
  if ( !handlers.empty() )
    error("Hander stack not empty. Something is wrong!");
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator>>(std::vector<uint>& x)
{
  message(1, "Reading array from file %s.", filename.c_str());
  XMLArray xml_array(x, *this);
  XMLDolfin xml_dolfin(xml_array, *this);
  xml_dolfin.handle();
  parse();
  if ( !handlers.empty() )
    error("Hander stack not empty. Something is wrong!");
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator>>(std::vector<double>& x)
{
  message(1, "Reading array from file %s.", filename.c_str());
  XMLArray xml_array(x, *this);
  XMLDolfin xml_dolfin(xml_array, *this);
  xml_dolfin.handle();
  parse();
  if ( !handlers.empty() )
    error("Hander stack not empty. Something is wrong!");
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator>>(std::map<uint, int>& map)
{
  message(1, "Reading map from file %s.", filename.c_str());
  XMLMap xml_map(map, *this);
  XMLDolfin xml_dolfin(xml_map, *this);
  xml_dolfin.handle();
  parse();
  if ( !handlers.empty() )
    error("Hander stack not empty. Something is wrong!");
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator>>(std::map<uint, uint>& map)
{
  message(1, "Reading map from file %s.", filename.c_str());
  XMLMap xml_map(map, *this);
  XMLDolfin xml_dolfin(xml_map, *this);
  xml_dolfin.handle();
  parse();
  if ( !handlers.empty() )
    error("Hander stack not empty. Something is wrong!");
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator>>(std::map<uint, double>& map)
{
  message(1, "Reading map from file %s.", filename.c_str());
  XMLMap xml_map(map, *this);
  XMLDolfin xml_dolfin(xml_map, *this);
  xml_dolfin.handle();
  parse();
  if ( !handlers.empty() )
    error("Hander stack not empty. Something is wrong!");
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator>>(std::map<uint, std::vector<int> >& array_map)
{
  message(1, "Reading array map from file %s.", filename.c_str());
  XMLMap xml_array_map(array_map, *this);
  XMLDolfin xml_dolfin(xml_array_map, *this);
  xml_dolfin.handle();
  parse();
  if ( !handlers.empty() )
    error("Hander stack not empty. Something is wrong!");
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator>>(std::map<uint, std::vector<uint> >& array_map)
{
  message(1, "Reading array map from file %s.", filename.c_str());
  XMLMap xml_array_map(array_map, *this);
  XMLDolfin xml_dolfin(xml_array_map, *this);
  xml_dolfin.handle();
  parse();
  if ( !handlers.empty() )
    error("Hander stack not empty. Something is wrong!");
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator>>(std::map<uint, std::vector<double> >& array_map)
{
  message(1, "Reading array map from file %s.", filename.c_str());
  XMLMap xml_array_map(array_map, *this);
  XMLDolfin xml_dolfin(xml_array_map, *this);
  xml_dolfin.handle();
  parse();
  if ( !handlers.empty() )
    error("Hander stack not empty. Something is wrong!");
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator<<(const std::vector<int>& x)
{
  open_file();
  XMLArray::write(x, *outstream, 1);
  close_file();
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator<<(const std::vector<uint>& x)
{
  open_file();
  XMLArray::write(x, *outstream, 1);
  close_file();
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator<<(const std::vector<double>& x)
{
  open_file();
  XMLArray::write(x, *outstream, 1);
  close_file();
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator<<(const std::map<uint, int>& map)
{
  open_file();
  XMLMap::write(map, *outstream, 1);
  close_file();
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator<<(const std::map<uint, uint>& map)
{
  open_file();
  XMLMap::write(map, *outstream, 1);
  close_file();
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator<<(const std::map<uint, double>& map)
{
  open_file();
  XMLMap::write(map, *outstream, 1);
  close_file();
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator<<(const std::map<uint, std::vector<int> >& array_map)
{
  open_file();
  XMLMap::write(array_map, *outstream, 1);
  close_file();
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator<<(const std::map<uint, std::vector<uint> >& array_map)
{
  open_file();
  XMLMap::write(array_map, *outstream, 1);
  close_file();
}
//-----------------------------------------------------------------------------
void NewXMLFile::operator<<(const std::map<uint, std::vector<double> >& array_map)
{
  open_file();
  XMLMap::write(array_map, *outstream, 1);
  close_file();
}
//-----------------------------------------------------------------------------
void NewXMLFile::parse()
{
  // Parse file
  xmlSAXUserParseFile(sax, (void *) this, filename.c_str());
}
//-----------------------------------------------------------------------------
void NewXMLFile::push(XMLHandler* handler)
{
  handlers.push(handler);
}
//-----------------------------------------------------------------------------
void NewXMLFile::pop()
{
  dolfin_assert( !handlers.empty() );
  handlers.pop();
}
//-----------------------------------------------------------------------------
XMLHandler* NewXMLFile:: top()
{
  dolfin_assert( !handlers.empty() );
  return handlers.top();
}
//-----------------------------------------------------------------------------
void NewXMLFile::start_element(const xmlChar *name, const xmlChar **attrs)
{
  handlers.top()->start_element(name, attrs);
}
//-----------------------------------------------------------------------------
void NewXMLFile::end_element(const xmlChar *name)
{
  handlers.top()->end_element(name);
}
//-----------------------------------------------------------------------------
void NewXMLFile::open_file()
{

  // Convert to ofstream
  std::ofstream* outfile = dynamic_cast<std::ofstream*>(outstream);
  if ( outfile != 0 )
  {
    // Open file
    outfile->open(filename.c_str());

    // Go to end of file
    outfile->seekp(0, std::ios::end);
  }
  XMLDolfin::write_start(*outstream);
}
//-----------------------------------------------------------------------------
void NewXMLFile::close_file()
{
  XMLDolfin::write_end(*outstream);
  // Convert to ofstream
  std::ofstream* outfile = dynamic_cast<std::ofstream*>(outstream);
  if ( outfile != 0 )
    outfile->close();

}
//-----------------------------------------------------------------------------
// Callback functions for the SAX interface
//-----------------------------------------------------------------------------
void dolfin::new_sax_start_document(void *ctx)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void dolfin::new_sax_end_document(void *ctx)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void dolfin::new_sax_start_element(void *ctx,
                                   const xmlChar *name,
                                   const xmlChar **attrs)
{
  ( (NewXMLFile*) ctx )->start_element(name, attrs);
}
//-----------------------------------------------------------------------------
void dolfin::new_sax_end_element(void *ctx, const xmlChar *name)
{
  ( (NewXMLFile*) ctx )->end_element(name);
}
//-----------------------------------------------------------------------------
void dolfin::new_sax_warning(void *ctx, const char *msg, ...)
{
  va_list args;
  va_start(args, msg);
  char buffer[DOLFIN_LINELENGTH];
  vsnprintf(buffer, DOLFIN_LINELENGTH, msg, args);
  warning("Incomplete XML data: " + std::string(buffer));
  va_end(args);
}
//-----------------------------------------------------------------------------
void dolfin::new_sax_error(void *ctx, const char *msg, ...)
{
  va_list args;
  va_start(args, msg);
  char buffer[DOLFIN_LINELENGTH];
  vsnprintf(buffer, DOLFIN_LINELENGTH, msg, args);
  error("Illegal XML data: " + std::string(buffer));
  va_end(args);
}
//-----------------------------------------------------------------------------
void dolfin::new_sax_fatal_error(void *ctx, const char *msg, ...)
{
  va_list args;
  va_start(args, msg);
  char buffer[DOLFIN_LINELENGTH];
  vsnprintf(buffer, DOLFIN_LINELENGTH, msg, args);
  error("Illegal XML data: " + std::string(buffer));
  va_end(args);
}
//-----------------------------------------------------------------------------
void dolfin::new_rng_parser_error(void *user_data, xmlErrorPtr error)
{
  char *file = error->file;
  char *message = error->message;
  int line = error->line;
  xmlNodePtr node;
  node = (xmlNode*)error->node;
  std::string buffer;
  buffer = message;
  int length = buffer.length();
  buffer.erase(length-1);
  if (node != NULL)
    warning("%s:%d: element %s: Relax-NG parser error: %s",
            file, line, node->name, buffer.c_str());
}
//-----------------------------------------------------------------------------
void dolfin::new_rng_valid_error(void *user_data, xmlErrorPtr error)
{
  char *file = error->file;
  char *message = error->message;
  int line = error->line;
  xmlNodePtr node;
  node = (xmlNode*)error->node;
  std::string buffer;
  buffer = message;
  int length = buffer.length();
  buffer.erase(length-1);
  warning("%s:%d: element %s: Relax-NG validity error: %s",
          file, line, node->name, buffer.c_str());
}
//-----------------------------------------------------------------------------

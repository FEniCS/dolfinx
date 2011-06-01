// Copyright (C) 2009 Ola Skavhaug
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2009.
//
// First added:  2009-03-03
// Last changed: 2011-03-31

#include <libxml/relaxng.h>

#include <dolfin/common/types.h>
#include <dolfin/common/constants.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/LocalMeshData.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/plot/FunctionPlotData.h>
#include "XMLArray.h"
#include "XMLMap.h"
#include "XMLFile.h"
#include "XMLMesh.h"
#include "XMLLocalMeshData.h"
#include "XMLMatrix.h"
#include "XMLFunctionPlotData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLFile::XMLFile(const std::string filename, bool gzip)
               : GenericFile(filename), sax(0), outstream(0), gzip(gzip)
{
  // Set up the output stream (to file)
  outstream = new std::ofstream();

  // Set up the sax handler.
  sax = new xmlSAXHandler();

  // Set up handlers for parser events
  sax->startDocument = sax_start_document;
  sax->endDocument   = sax_end_document;
  sax->startElement  = sax_start_element;
  sax->endElement    = sax_end_element;
  sax->warning       = sax_warning;
  sax->error         = sax_error;
  sax->fatalError    = sax_fatal_error;
}
//-----------------------------------------------------------------------------
XMLFile::XMLFile(std::ostream& s) : GenericFile(""), sax(0), outstream(&s)
{
  // Set up the sax handler.
  sax = new xmlSAXHandler();

  // Set up handlers for parser events
  sax->startDocument = sax_start_document;
  sax->endDocument   = sax_end_document;
  sax->startElement  = sax_start_element;
  sax->endElement    = sax_end_element;
  sax->warning       = sax_warning;
  sax->error         = sax_error;
  sax->fatalError    = sax_fatal_error;
}
//-----------------------------------------------------------------------------
XMLFile::~XMLFile()
{
  delete sax;

  // Only delete outstream if it is an 'ofstream'
  std::ofstream* outfile = dynamic_cast<std::ofstream*>(outstream);
  if (outfile)
  {
    outfile = 0;
    delete outstream;
  }
}
//-----------------------------------------------------------------------------
void XMLFile::validate(const std::string filename)
{
  xmlRelaxNGParserCtxtPtr parser;
  xmlRelaxNGValidCtxtPtr validator;
  xmlRelaxNGPtr schema;
  xmlDocPtr document;
  document = xmlParseFile(filename.c_str());
  int ret = 1;
  parser = xmlRelaxNGNewParserCtxt("http://fenicsproject.org/pub/misc/dolfin.rng");
  xmlRelaxNGSetParserStructuredErrors(parser,
                                      (xmlStructuredErrorFunc)rng_parser_error,
                                      stderr);
  schema = xmlRelaxNGParse(parser);
  validator = xmlRelaxNGNewValidCtxt(schema);
  xmlRelaxNGSetValidStructuredErrors(validator,
                                     (xmlStructuredErrorFunc)rng_valid_error,
                                     stderr);
  ret = xmlRelaxNGValidateDoc(validator, document);
  if (ret == 0)
    log(DBG, "%s validates", filename.c_str());
  else if ( ret < 0 )
    error("%s failed to load", filename.c_str());
  else
    error("%s fails to validate", filename.c_str());

  xmlRelaxNGFreeValidCtxt(validator);
}
//-----------------------------------------------------------------------------
void XMLFile::write()
{
  if (gzip)
    error("Unable to write XML data, gzipped XML (.xml.gz) not supported for output.");
}
//-----------------------------------------------------------------------------
void XMLFile::parse()
{
  // Parse file
  xmlSAXUserParseFile(sax, (void *) this, filename.c_str());
}
//-----------------------------------------------------------------------------
void XMLFile::push(XMLHandler* handler)
{
  handlers.push(handler);
}
//-----------------------------------------------------------------------------
void XMLFile::pop()
{
  assert(!handlers.empty());
  handlers.pop();
}
//-----------------------------------------------------------------------------
XMLHandler* XMLFile:: top()
{
  assert(!handlers.empty());
  return handlers.top();
}
//-----------------------------------------------------------------------------
void XMLFile::start_element(const xmlChar *name, const xmlChar **attrs)
{
  handlers.top()->start_element(name, attrs);
}
//-----------------------------------------------------------------------------
void XMLFile::end_element(const xmlChar *name)
{
  handlers.top()->end_element(name);
}
//-----------------------------------------------------------------------------
void XMLFile::open_file()
{
  // Convert to ofstream
  std::ofstream* outfile = dynamic_cast<std::ofstream*>(outstream);
  if (outfile)
  {
    // Open file
    outfile->open(filename.c_str());

    // Go to end of file
    outfile->seekp(0, std::ios::end);
  }
  XMLDolfin::write_start(*outstream);
}
//-----------------------------------------------------------------------------
void XMLFile::close_file()
{
  XMLDolfin::write_end(*outstream);

  // Convert to ofstream
  std::ofstream* outfile = dynamic_cast<std::ofstream*>(outstream);
  if (outfile)
    outfile->close();
}
//-----------------------------------------------------------------------------
// Callback functions for the SAX interface
//-----------------------------------------------------------------------------
void dolfin::sax_start_document(void *ctx)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void dolfin::sax_end_document(void *ctx)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void dolfin::sax_start_element(void *ctx, const xmlChar *name,
                               const xmlChar **attrs)
{
  ( (XMLFile*) ctx )->start_element(name, attrs);
}
//-----------------------------------------------------------------------------
void dolfin::sax_end_element(void *ctx, const xmlChar *name)
{
  ( (XMLFile*) ctx )->end_element(name);
}
//-----------------------------------------------------------------------------
void dolfin::sax_warning(void *ctx, const char *msg, ...)
{
  va_list args;
  va_start(args, msg);
  char buffer[DOLFIN_LINELENGTH];
  vsnprintf(buffer, DOLFIN_LINELENGTH, msg, args);
  warning("Incomplete XML data: " + std::string(buffer));
  va_end(args);
}
//-----------------------------------------------------------------------------
void dolfin::sax_error(void *ctx, const char *msg, ...)
{
  va_list args;
  va_start(args, msg);
  char buffer[DOLFIN_LINELENGTH];
  vsnprintf(buffer, DOLFIN_LINELENGTH, msg, args);
  error("Illegal XML data: " + std::string(buffer));
  va_end(args);
}
//-----------------------------------------------------------------------------
void dolfin::sax_fatal_error(void *ctx, const char *msg, ...)
{
  va_list args;
  va_start(args, msg);
  char buffer[DOLFIN_LINELENGTH];
  vsnprintf(buffer, DOLFIN_LINELENGTH, msg, args);
  error("Illegal XML data: " + std::string(buffer));
  va_end(args);
}
//-----------------------------------------------------------------------------
void dolfin::rng_parser_error(void *user_data, xmlErrorPtr error)
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
  {
    warning("%s:%d: element %s: Relax-NG parser error: %s",
            file, line, node->name, buffer.c_str());
  }
}
//-----------------------------------------------------------------------------
void dolfin::rng_valid_error(void *user_data, xmlErrorPtr error)
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

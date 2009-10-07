// Copyright (C) 2009 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-16
// Last changed: 2009-10-07

#include <dolfin/io/XMLFile.h>
#include <dolfin/plot/FunctionPlotData.h>
#include "XMLIndent.h"
#include "XMLMesh.h"
#include "XMLFunctionPlotData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLFunctionPlotData::XMLFunctionPlotData(FunctionPlotData& data, XMLFile& parser)
  : XMLHandler(parser), data(data), state(OUTSIDE), xml_mesh(0), xml_vector(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
XMLFunctionPlotData::~XMLFunctionPlotData()
{
  delete xml_mesh;
}
//-----------------------------------------------------------------------------
void XMLFunctionPlotData::start_element(const xmlChar* name, const xmlChar** attrs)
{
  switch ( state )
  {
  case OUTSIDE:
    if ( xmlStrcasecmp(name, (xmlChar *) "function_plot_data") == 0 )
      read_data_tag(name, attrs);

    break;

  case INSIDE:
    if ( xmlStrcasecmp(name, (xmlChar *) "mesh") == 0 )
      read_mesh(name, attrs);
    else if( xmlStrcasecmp(name, (xmlChar *) "vector") == 0 )
      read_vector(name, attrs);

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLFunctionPlotData::end_element(const xmlChar* name)
{
  switch ( state )
  {
  case INSIDE:
    if ( xmlStrcasecmp(name, (xmlChar *) "function_plot_data") == 0 )
    {
      state = DONE;
      release();
    }

    break;

  default:
    ;
  }
}
//-----------------------------------------------------------------------------
void XMLFunctionPlotData::write(const FunctionPlotData& data,
                                std::ostream& outfile, uint indentation_level)
{
  XMLIndent indent(indentation_level);

  // Write Function plot data header
  outfile << indent();
  outfile << "<function_plot_data rank=\"" << data.rank << "\">" << std::endl;

  ++indent;

  // Write mesh
  XMLMesh::write(data.mesh, outfile, indent.level());

  // Write vector
  XMLVector::write(data.vertex_values(), outfile, indent.level());

  --indent;

  // Write Function plot data footer
  outfile << indent() << "</function_plot_data>" << std::endl;
}
//-----------------------------------------------------------------------------
void XMLFunctionPlotData::read_data_tag(const xmlChar* name, const xmlChar** attrs)
{
  state = INSIDE;
  data.rank = parse_uint(name, attrs, "rank");
}
//-----------------------------------------------------------------------------
void XMLFunctionPlotData::read_mesh(const xmlChar* name, const xmlChar** attrs)
{
  delete xml_mesh;
  xml_mesh = new XMLMesh(data.mesh, parser);

  // Let the xml mesh read its own the mesh tag
  xml_mesh->read_mesh_tag(name, attrs);

  // Parse the rest of the mesh
  xml_mesh->handle();
}
//-----------------------------------------------------------------------------
void XMLFunctionPlotData::read_vector(const xmlChar* name, const xmlChar** attrs)
{
  delete xml_vector;
  xml_vector = new XMLVector(data.vertex_values(), parser);

  // Let the xml vector read its own the vector tag
  xml_vector->read_vector_tag(name, attrs);

  // Parse the rest of the vector
  xml_vector->handle();
}
//-----------------------------------------------------------------------------

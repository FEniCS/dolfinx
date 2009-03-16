// Copyright (C) 2009 Ola Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-16
// Last changed: 2009-03-16

#include <dolfin/io/NewXMLFile.h>
#include <dolfin/plot/FunctionPlotData.h>
#include "NewXMLMesh.h"
#include "XMLFunctionPlotData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
XMLFunctionPlotData::XMLFunctionPlotData(FunctionPlotData& data, NewXMLFile& parser)
  : XMLHandler(parser), data(data), state(OUTSIDE), xml_mesh(0)
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
void XMLFunctionPlotData::write(const FunctionPlotData& data, std::ostream& outfile, uint indentation_level)
{
  // Write Function plot data header
  uint curr_indent = indentation_level;
  outfile << std::setw(curr_indent) << "";
  outfile << "<function_plot_data rank=\"" << data.rank << "\">" << std::endl;

  curr_indent  = indentation_level + 2;

  // Write mesh
  NewXMLMesh::write(data.mesh, outfile, curr_indent);

  // Write vector
  NewXMLVector::write(data.vertex_values, outfile, curr_indent);

  // Write Function plot data footer
  curr_indent = indentation_level;
  outfile << std::setw(curr_indent) << "";
  outfile << "</function_plot_data>" << std::endl;
  //Write me later
}
//-----------------------------------------------------------------------------
void XMLFunctionPlotData::read_data_tag(const xmlChar* name, const xmlChar** attrs)
{
  data.rank = parse_uint(name, attrs, "rank");
  state = INSIDE;
}
//-----------------------------------------------------------------------------
void XMLFunctionPlotData::read_mesh(const xmlChar* name, const xmlChar** attrs)
{
  delete xml_mesh;
  xml_mesh = new NewXMLMesh(data.mesh, parser);
  xml_mesh->read_mesh_tag(name, attrs);
  xml_mesh->handle();
}
//-----------------------------------------------------------------------------
void XMLFunctionPlotData::read_vector(const xmlChar* name, const xmlChar** attrs)
{

}

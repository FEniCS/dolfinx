// Copyright (C) 2002-2011 Anders Logg, Ola Skavhaug and Garth N. Wells
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2002-12-06
// Last changed: 2011-06-21

#include "pugixml.hpp"

#include "dolfin/la/GenericVector.h"
#include "dolfin/log/log.h"
#include "dolfin/mesh/Mesh.h"
#include "dolfin/plot/FunctionPlotData.h"
#include "XMLIndent.h"
#include "XMLMesh.h"
#include "XMLVector.h"
#include "XMLFunctionPlotData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void XMLFunctionPlotData::read(FunctionPlotData& plot_data,
                               const pugi::xml_node xml_mesh)
{
  const pugi::xml_node xml_plot_node = xml_mesh.child("function_plot_data");
  if (!xml_plot_node)
    error("Not a DOLFIN FunctionPlotData file.");

  // Get rank
  const unsigned int rank = xml_plot_node.attribute("rank").as_uint();
  plot_data.rank = rank;

  // Read mesh
  Mesh& mesh = plot_data.mesh;
  XMLMesh::read(mesh, xml_plot_node);

  // Read vector
  GenericVector& vector = plot_data.vertex_values();
  XMLVector::read(vector, xml_plot_node);
}
//-----------------------------------------------------------------------------
void XMLFunctionPlotData::write(const FunctionPlotData& plot_data,
                                std::ostream& outfile,
                                uint indentation_level)
{
  XMLIndent indent(indentation_level);

  // Write Function plot data header
  outfile << indent();
  outfile << "<function_plot_data rank=\"" << plot_data.rank << "\">" << std::endl;

  ++indent;

  // Write mesh
  XMLMesh::write(plot_data.mesh, outfile, indent.level());

  // Write vector
  XMLVector::write(plot_data.vertex_values(), outfile, indent.level());

  --indent;

  // Write Function plot data footer
  outfile << indent() << "</function_plot_data>" << std::endl;
}
//-----------------------------------------------------------------------------

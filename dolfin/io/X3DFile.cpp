// Copyright (C) 2013 Chris Richardson
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
// First added:  2013-03-05
// Last changed: 2013-05-10

#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>

#include "pugixml.hpp"

#include <dolfin/common/MPI.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Face.h>
#include <dolfin/mesh/Vertex.h>

#include "X3DFile.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
X3DFile::X3DFile(const std::string filename) : GenericFile(filename, "X3D"),
  facet_type("IndexedFaceSet")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
X3DFile::~X3DFile()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void X3DFile::operator<< (const Mesh& mesh)
{
  write_mesh(mesh);
}
//-----------------------------------------------------------------------------
void X3DFile::operator<< (const MeshFunction<std::size_t>& meshfunction)
{
  write_meshfunction(meshfunction);
}
//-----------------------------------------------------------------------------
void X3DFile::operator<< (const Function& u)
{
  write_function(u);
}
//-----------------------------------------------------------------------------
std::string X3DFile::color_palette(const int palette) const
{
  // Make a basic palette of 256 colours
  std::stringstream colour;
  switch (palette)
  {
  case 1:
    for (int i = 0; i < 256; ++i)
    {
      const double x = (double)i/255.0;
      const double y = 1.0 - x;
      const double r = 4*pow(x, 3) - 3*pow(x, 4);
      const double g = 4*pow(x, 2)*(1.0 - pow(x, 2));
      const double b = 4*pow(y, 3) - 3*pow(y, 4);
      colour << r << " " << g << " " << b << " ";
    }
    break;

  case 2:
    for (int i = 0; i < 256; ++i)
    {
      const double lm = 425.0 + 250.0*(double)i/255.0;
      const double b = 1.8*exp(-pow((lm - 450.0)/((lm>450.0) ? 40.0 : 20.0), 2.0));
      const double g = 0.9*exp(-pow((lm - 550.0)/((lm>550.0) ? 60 : 40.0), 2.0));
      double r = 1.0*exp(-pow((lm - 600.0)/((lm>600.0) ? 40.0 : 50.0), 2.0));
      r += 0.3*exp(-pow((lm - 450.0)/((lm>450.0) ? 20.0 : 30.0), 2.0));
      const double tot = (r + g + b);

      colour << r/tot << " " << g/tot << " " << b/tot << " ";
    }
    break;

  default:
    for (int i = 0; i < 256 ; ++i)
    {
      const double r = (double)i/255.0;
      const double g = (double)(i*(255 - i))/(128.0*127.0);
      const double b = (double)(255 - i)/255.0;
      colour << r << " " << g << " " << b << " ";
    }
    break;
  }

  return colour.str();
}
//-----------------------------------------------------------------------------
void X3DFile::write_meshfunction(const MeshFunction<std::size_t>& meshfunction)
{
  // Palette choice
  const int palette = 2;

  // Get mesh
  dolfin_assert( meshfunction.mesh());
  const Mesh& mesh = *meshfunction.mesh();

  // Ensure connectivity has been computed
  mesh.init(mesh.topology().dim() - 1 , mesh.topology().dim());

  // Mesh geometric dimension
  const std::size_t gdim = mesh.geometry().dim();

  // Mesh topological dimension
  const std::size_t tdim = mesh.topology().dim();

  // MeshFunction dimension
  const std::size_t cell_dim = meshfunction.dim();

  // Check that MeshFunction dimension is handled
  if (cell_dim != tdim)
  {
    dolfin_error("X3DFile.cpp",
                 "output meshfunction",
                 "Can only output CellFunction at present");
  }

  // Check that X3D type is appropriate
  if (cell_dim == tdim && facet_type == "IndexedLineSet")
  {
    dolfin_error("X3DFile.cpp",
                 "output meshfunction",
                 "Cannot output CellFunction with Edge mesh");
  }

  // Check that mesh is in 2D or 3D
  if (gdim !=2 && gdim !=3)
  {
    dolfin_error("X3DFile.cpp",
                 "output mesh",
                 "X3D will only output 2D or 3D meshes");
  }

  // Pointer to MeshFunction data
  const std::size_t* values = meshfunction.values();

  // Get min/max values of MeshFunction
  std::size_t minval = *std::min_element(values, values + meshfunction.size());
  minval = MPI::min(minval);
  std::size_t maxval = *std::max_element(values, values + meshfunction.size());
  maxval = MPI::max(maxval);
  double dval;
  if (maxval == minval)
    dval = 1.0;
  else
    dval = 255.0/(double)(maxval - minval);

  // Get mesh min/max  dimensions and viewpoint
  const std::vector<double> xpos = mesh_min_max(mesh);

  // Get MPI details
  const std::size_t num_processes = MPI::num_processes();
  const std::size_t process_number = MPI::process_number();

  // Create pugi xml document
  pugi::xml_document xml_doc;

  // Write XML header
  output_xml_header(xml_doc, xpos);

  // Make a set of the indices we wish to use. In 3D, we are ignoring
  // all interior facets, so reducing the number of vertices
  // substantially
  const std::vector<std::size_t> vecindex = vertex_index(mesh);

  // Write vertices
  write_vertices(xml_doc, mesh, vecindex);

  // Iterate over mesh facets
  std::stringstream local_output;
  for (FaceIterator f(mesh); !f.end(); ++f)
  {
    // Check if topolgical dimension is 2, or if we have a boundary
    // facet in 3D
    if (tdim == 2 || f->num_global_entities(tdim) == 1)
    {
      // Get cell connected to facet
      CellIterator cell(*f);

      // Write mesh function value to string stream
      local_output << (int)((meshfunction[*cell] - minval)*dval) << " " ;
    }
  }

  // Gather up data on zero
  std::vector<std::string> gathered_output;
  MPI::gather(local_output.str(), gathered_output);

  // Write XML on root process
  if (process_number == 0)
  {
    pugi::xml_node indexed_face_set = xml_doc.child("X3D")
      .child("Scene").child("Shape").child(facet_type.c_str());
    indexed_face_set.append_attribute("colorPerVertex") = "false";

    std::stringstream str_output;
    for(std::size_t i = 0; i < num_processes; ++i)
      str_output << gathered_output[i];
    indexed_face_set.append_attribute("colorIndex") = str_output.str().c_str();

    // Output palette
    pugi::xml_node color = indexed_face_set.append_child("Color");
    color.append_attribute("color") = color_palette(palette).c_str();

    // Save XML file
    xml_doc.save_file(_filename.c_str(), "  ");
  }
}
//-----------------------------------------------------------------------------
void X3DFile::write_function(const Function& u)
{
  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();

  // Mesh geometric and tological dimensions
  const std::size_t gdim = mesh.geometry().dim();
  const std::size_t tdim = mesh.topology().dim();

  if (gdim !=2 && gdim !=3)
  {
    dolfin_error("X3DFile.cpp",
                 "output mesh",
                 "X3D will only output 2D or 3D meshes");
  }

  // Build mesh connectivity
  mesh.init(tdim - 1 , tdim);

  dolfin_assert(u.function_space()->dofmap());
  const GenericDofMap& dofmap = *u.function_space()->dofmap();

  // Only allow scalar fields
  if (u.value_rank() > 1)
  {
    dolfin_error("X3DFile.cpp",
                 "write X3D",
                 "Can only handle scalar and vector Functions");
  }

  if (u.value_rank() == 1)
    warning("X3DFile outputs scalar magnitude of vector field");

  // Only allow vertex centered data
  const bool vertex_data = (dofmap.max_cell_dimension() != 1);
  if (!vertex_data)
  {
    dolfin_error("X3DFile.cpp",
                 "write X3D",
                 "Can only handle vertex-based Function at present");
  }

  // Compute vertex data values
  std::vector<double> data_values;
  u.compute_vertex_values(data_values, mesh);

  // Normalise data values
  if (u.value_rank() == 1)
  {
    std::vector<double> magnitude;
    const std::size_t num_vertices = mesh.num_vertices();
    for (std::size_t i = 0; i < num_vertices ; ++i)
    {
      double val = 0.0;
      for(std::size_t j = 0; j < u.value_size() ; j++)
        val += pow(data_values[i + j*num_vertices], 2.0);
      val = sqrt(val);
      magnitude.push_back(val);
    }
    data_values.resize(magnitude.size());
    std::copy(magnitude.begin(), magnitude.end(), data_values.begin());
  }

  // Create pugi document
  pugi::xml_document xml_doc;

  // Get mesh mix/max dimensions and wriet XML header
  const std::vector<double> xpos = mesh_min_max(mesh);
  output_xml_header(xml_doc, xpos);

  // Get indices of vertices on mesh surface
  const std::vector<std::size_t> surface_vertices = vertex_index(mesh);

  // Write vertices and vertex data to XML file
  write_vertices(xml_doc, mesh, surface_vertices);
  write_values(xml_doc, mesh, surface_vertices, data_values);

  // Save XML file
  if (MPI::process_number() == 0)
    xml_doc.save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
void X3DFile::write_mesh(const Mesh& mesh)
{
  const std::size_t gdim = mesh.geometry().dim();
  if (gdim !=2 && gdim !=3)
  {
    dolfin_error("X3DFile.cpp",
                 "output mesh",
                 "X3D will only output 2D or 3D meshes");
  }

  // Creat pugi doc
  pugi::xml_document xml_doc;

  // For serial - ensure connectivity
  mesh.init(mesh.topology().dim() - 1 , mesh.topology().dim());

  // Get mesh max and min dimensions and viewpoint
  const std::vector<double> xpos = mesh_min_max(mesh);

  // Create XML for all mesh vertices on surface
  output_xml_header(xml_doc, xpos);
  const std::vector<std::size_t> vecindex = vertex_index(mesh);
  write_vertices(xml_doc, mesh, vecindex);

  if (MPI::process_number() == 0)
    xml_doc.save_file(_filename.c_str(), "  ");
}
//-----------------------------------------------------------------------------
void X3DFile::write_values(pugi::xml_document& xml_doc, const Mesh& mesh,
                           const std::vector<std::size_t> vecindex,
                           const std::vector<double> data_values)
{
  const std::size_t tdim = mesh.topology().dim();

  double minval = *std::min_element(data_values.begin(), data_values.end());
  minval = MPI::min(minval);
  double maxval = *std::max_element(data_values.begin(), data_values.end());
  maxval = MPI::max(maxval);

  double scale = 0.0;
  if (maxval == minval)
    scale = 1.0;
  else
    scale = 255.0/(maxval - minval);

  std::stringstream local_output;
  if (facet_type == "IndexedLineSet")
  {
    for (EdgeIterator e(mesh); !e.end(); ++e)
    {
      bool allow_edge = (tdim == 2);

      // If one of the faces connected to this edge is external, then
      // output the edge
      if (!allow_edge)
      {
        for(FaceIterator f(*e); !f.end(); ++f)
        {
          if (f->num_global_entities(tdim) == 1)
            allow_edge = true;
        }
      }

      if (allow_edge)
      {
        for (VertexIterator v(*e); !v.end(); ++v)
        {
          local_output << (int)((data_values[v->index()] - minval)*scale)
                       << " " ;
        }
        local_output << "-1 ";
      }
    }
  }
  else if (facet_type == "IndexedFaceSet")
  {
    // Output faces
    for (FaceIterator f(mesh); !f.end(); ++f)
    {
      if (tdim == 2 || f->num_global_entities(tdim) == 1)
      {
        for (VertexIterator v(*f); !v.end(); ++v)
        {
          local_output << (int)((data_values[v->index()] - minval)*scale)
                       << " " ;
        }
        local_output << "-1 ";
      }
    }
  }

  // Gather up on zero
  std::vector<std::string> gathered_output;
  MPI::gather(local_output.str(), gathered_output);
  if (MPI::process_number() == 0)
  {
    pugi::xml_node indexed_face_set = xml_doc.child("X3D")
      .child("Scene").child("Shape").child(facet_type.c_str());
    indexed_face_set.append_attribute("colorPerVertex") = "true";

    std::stringstream str_output;
    for(std::size_t i = 0; i < MPI::num_processes(); ++i)
      str_output << gathered_output[i];

    indexed_face_set.append_attribute("colorIndex") = str_output.str().c_str();

    // Output colour palette
    const int palette = 2;
    pugi::xml_node color = indexed_face_set.append_child("Color");
    color.append_attribute("color") = color_palette(palette).c_str();
  }
}
//-----------------------------------------------------------------------------
void X3DFile::write_vertices(pugi::xml_document& xml_doc, const Mesh& mesh,
                             const std::vector<std::size_t> vecindex)
{
  std::size_t offset = MPI::global_offset(vecindex.size(), true);

  const std::size_t process_number = MPI::process_number();
  const std::size_t num_processes = MPI::num_processes();
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t gdim = mesh.geometry().dim();

  std::stringstream local_output;
  if (facet_type == "IndexedLineSet")
  {
    for (EdgeIterator e(mesh); !e.end(); ++e)
    {
      bool allow_edge = (tdim == 2);

      // If one of the faces connected to this edge is external, then
      // output the edge
      if (!allow_edge)
      {
        for(FaceIterator f(*e); !f.end(); ++f)
        {
          if (f->num_global_entities(tdim) == 1)
            allow_edge = true;
        }
      }

      if (allow_edge)
      {
        for(VertexIterator v(*e); !v.end(); ++v)
        {
          std::size_t index_it  = std::find(vecindex.begin(),
                                            vecindex.end(),
                                            v->index()) - vecindex.begin();
          local_output << index_it + offset  << " " ;
        }
        local_output << "-1 ";
      }
    }
  }
  else if (facet_type == "IndexedFaceSet")
  {
    // Output faces
    for (FaceIterator f(mesh); !f.end(); ++f)
    {
      if (tdim == 2 || f->num_global_entities(tdim) == 1)
      {
        for (VertexIterator v(*f); !v.end(); ++v)
        {
          std::size_t index_it = std::find(vecindex.begin(),
                                           vecindex.end(),
                                           v->index()) - vecindex.begin();
          local_output << index_it + offset  << " " ;
        }
        local_output << "-1 ";
      }
    }
  }

  std::vector<std::string> gathered_output;
  MPI::gather(local_output.str(), gathered_output);

  if (process_number == 0)
  {
    pugi::xml_node indexed_face_set
      = xml_doc.child("X3D").child("Scene").child("Shape").child(facet_type.c_str());

    std::stringstream str_output;
    for (std::size_t i = 0; i < num_processes; ++i)
      str_output << gathered_output[i];

    indexed_face_set.append_attribute("coordIndex") = str_output.str().c_str();
  }

  local_output.str("");

  // Now fill in the geometry
  for (std::vector<std::size_t>::const_iterator index = vecindex.begin();
       index != vecindex.end(); ++index)
  {
    Vertex v(mesh, *index);
    local_output << v.x(0) << " " << v.x(1) << " ";
    if (gdim==2)
      local_output << "0 ";
    else
      local_output << v.x(2) << " ";
  }

  MPI::gather(local_output.str(), gathered_output);

  // Finally, close off with the XML footer on process zero
  if (process_number == 0)
  {
    pugi::xml_node indexed_face_set
      = xml_doc.child("X3D").child("Scene").child("Shape").child(facet_type.c_str());

    pugi::xml_node coordinate = indexed_face_set.append_child("Coordinate");

    std::stringstream str_output;
    for(std::size_t i = 0; i < num_processes; ++i)
      str_output << gathered_output[i];

    coordinate.append_attribute("point") = str_output.str().c_str();
  }
}
//-----------------------------------------------------------------------------
void X3DFile::output_xml_header(pugi::xml_document& xml_doc,
                                const std::vector<double>& xpos)
{
  if (MPI::process_number() == 0)
  {
    xml_doc.append_child(pugi::node_doctype).set_value("X3D PUBLIC \"ISO//Web3D//DTD X3D 3.2//EN\" \"http://www.web3d.org/specifications/x3d-3.2.dtd\"");

    pugi::xml_node x3d = xml_doc.append_child("X3D");
    x3d.append_attribute("profile") = "Interchange";
    x3d.append_attribute("version") = "3.2";
    x3d.append_attribute("xmlns:xsd")
      = "http://www.w3.org/2001/XMLSchema-instance";
    x3d.append_attribute("xsd:noNamespaceSchemaLocation")
      = "http://www.web3d.org/specifications/x3d-3.2.xsd";

    pugi::xml_node scene = x3d.append_child("Scene");
    pugi::xml_node viewpoint = scene.append_child("Viewpoint");
    std::string xyz = boost::lexical_cast<std::string>(xpos[0]) + " "
      + boost::lexical_cast<std::string>(xpos[1]) + " "
      + boost::lexical_cast<std::string>(xpos[3]);
    viewpoint.append_attribute("position") = xyz.c_str();

    viewpoint.append_attribute("orientation") = "0 0 0 1";
    viewpoint.append_attribute("fieldOfView") = "0.785398";
    xyz = boost::lexical_cast<std::string>(xpos[0]) + " "
      + boost::lexical_cast<std::string>(xpos[1]) + " "
      + boost::lexical_cast<std::string>(xpos[2]);
    viewpoint.append_attribute("centerOfRotation") = xyz.c_str();

    viewpoint.append_attribute("zNear") = "-1";
    viewpoint.append_attribute("zFar") = "-1";
    pugi::xml_node background = scene.append_child("Background");
    background.append_attribute("skyColor") = "0.9 0.9 1.0";

    pugi::xml_node shape = scene.append_child("Shape");
    pugi::xml_node material
      = shape.append_child("Appearance").append_child("Material");
    material.append_attribute("ambientIntensity") = "0.05";
    material.append_attribute("shininess") = "0.5";
    material.append_attribute("diffuseColor") = "0.7 0.7 0.7";
    material.append_attribute("specularColor") = "0.9 0.9 0.9";
    material.append_attribute("emmisiveColor") = "0.7 0.7 0.7";

    shape.append_child(facet_type.c_str()).append_attribute("solid") = "false";
  }
}
//-----------------------------------------------------------------------------
std::vector<double> X3DFile::mesh_min_max(const Mesh& mesh) const
{
  // Get dimensions
  double xmin = std::numeric_limits<double>::max();
  double xmax = std::numeric_limits<double>::min();
  double ymin = std::numeric_limits<double>::max();
  double ymax = std::numeric_limits<double>::min();
  double zmin = std::numeric_limits<double>::max();
  double zmax = std::numeric_limits<double>::min();

  const std::size_t gdim = mesh.geometry().dim();

  for (VertexIterator v(mesh); !v.end(); ++v)
  {
    xmin = std::min(xmin, v->x(0));
    xmax = std::max(xmax, v->x(0));
    ymin = std::min(ymin, v->x(1));
    ymax = std::max(ymax, v->x(1));
    if (gdim == 2)
    {
      zmin = 0.0;
      zmax = 0.0;
    }
    else
    {
      zmin = std::min(zmin, v->x(2));
      zmax = std::max(zmax, v->x(2));
    }
  }

  xmin = MPI::min(xmin);
  ymin = MPI::min(ymin);
  zmin = MPI::min(zmin);

  xmax = MPI::max(xmax);
  ymax = MPI::max(ymax);
  zmax = MPI::max(zmax);

  std::vector<double> result;
  result.push_back((xmax + xmin)/2.0);
  result.push_back((ymax + zmin)/2.0);
  result.push_back((zmax + zmin)/2.0);

  double d = std::max(xmax - xmin, ymax - ymin);
  d = 2.0*std::max(d, zmax - zmin) + zmax ;
  result.push_back(d);

  return result;
}
//-----------------------------------------------------------------------------
std::vector<std::size_t> X3DFile::vertex_index(const Mesh& mesh) const
{
  const std::size_t tdim = mesh.topology().dim();
  std::set<std::size_t> vindex;

  for (FaceIterator f(mesh); !f.end(); ++f)
  {
    // If in 3D, only output exterior faces
    // FIXME: num_global_entities not working in serial
    if (tdim == 2 || f->num_global_entities(tdim) == 1)
    {
      for(VertexIterator v(*f); !v.end(); ++v)
        vindex.insert(v->index());
    }
  }

  // Copy to vector for wider iterator support
  const std::vector<std::size_t> vecindex(vindex.begin(), vindex.end());
  return vecindex;
}
//-----------------------------------------------------------------------------

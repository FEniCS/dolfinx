// Copyright (C) 2016 Quang T. Ha, Chris Richardson and Garth N. Wells
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

#include <sstream>
#include <boost/lexical_cast.hpp>

#include <dolfin/common/MPI.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Face.h>
#include <dolfin/mesh/Vertex.h>

#include "X3DOM.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::string X3DOM::str(const Mesh& mesh, Representation facet_type,
                      Viewpoints viewpoint_switch)
{
  // Create empty pugi XML doc
  pugi::xml_document xml_doc;

  // Build X3D XML and add to XML doc
  x3dom_xml(xml_doc, mesh, facet_type, viewpoint_switch);

  // Save XML doc to stringstream
  std::stringstream s;
  const std::string indent = "  ";
  xml_doc.save(s, indent.c_str());

  // Return string
  return s.str();
}
//-----------------------------------------------------------------------------
std::string X3DOM::html(const Mesh& mesh, Representation facet_type)
{
  return html(mesh, facet_type, Viewpoints::On);
}
//-----------------------------------------------------------------------------
std::string X3DOM::html(const Mesh& mesh, Representation facet_type, 
                            Viewpoints viewpoint_switch)
{
  // Create empty pugi XML doc
  pugi::xml_document xml_doc;

  // Add html node
  pugi::xml_node node = xml_doc.append_child("html");

  // Add head node
  pugi::xml_node head = node.append_child("head");

  // Add script node
  pugi::xml_node script = head.append_child("script");

  // Set attributes for script node
  script.append_attribute("type") = "text/javascript";
  script.append_attribute("src") = "http://www.x3dom.org/download/x3dom.js";
  script.append_child(pugi::node_pcdata);

  // Add link node
  pugi::xml_node link = head.append_child("link");

  // Set attributes for link node
  link.append_attribute("rel") = "stylesheet";
  link.append_attribute("type") = "text/css";
  link.append_attribute("href") = "http://www.x3dom.org/download/x3dom.css";

  // Add body node
  pugi::xml_node body_node = node.append_child("body");

  // Add X3D XML data to 'body' node
  x3dom_xml(body_node, mesh, facet_type, viewpoint_switch);

  // Now append four viewpoints
  // FIXME Write Function to do this
  if (viewpoint_switch == Viewpoints::On)
  {
    // Add viewpoint control node
    pugi::xml_node viewpoint_control = body_node.append_child("div");

    // Add attributes to viewpoint niode
    viewpoint_control.append_attribute("id") = "camera_buttons";
    viewpoint_control.append_attribute("style") = "display: block";

    add_viewpoint_control_option(viewpoint_control, "Front");
    add_viewpoint_control_option(viewpoint_control, "Back");
    add_viewpoint_control_option(viewpoint_control, "Left");
    add_viewpoint_control_option(viewpoint_control, "Right");
    add_viewpoint_control_option(viewpoint_control, "Top");
    add_viewpoint_control_option(viewpoint_control, "Bottom");
  }

  // Save XML doc to stringstream, without default XML header
  std::stringstream s;
  const std::string indent = "  ";
  xml_doc.save(s, indent.c_str(), pugi::format_default | pugi::format_no_declaration);

  // Return string
  return s.str();
}
//-----------------------------------------------------------------------------
void X3DOM::add_viewpoint_control_option(pugi::xml_node& viewpoint_control, std::string vp)
{
  std::string onclick_str = "document.getElementById('" + vp + "').setAttribute('set_bind','true');";
  pugi::xml_node viewpoint_option = viewpoint_control.append_child("button");
  viewpoint_option.append_attribute("onclick") = onclick_str.c_str();
  viewpoint_option.append_attribute("style") = "display: block";
  viewpoint_option.append_child(pugi::node_pcdata).set_value(vp.c_str());
}
//-----------------------------------------------------------------------------
/*
std::string X3DOM::str(const MeshFunction<std::size_t>& meshfunction,
                       const std::string facet_type, const size_t palette)
{
  // Get mesh
  dolfin_assert(meshfunction.mesh());
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
  minval = MPI::min(mesh.mpi_comm(), minval);
  std::size_t maxval = *std::max_element(values, values + meshfunction.size());
  maxval = MPI::max(mesh.mpi_comm(), maxval);
  double dval;
  if (maxval == minval)
    dval = 1.0;
  else
    dval = 255.0/(double)(maxval - minval);

  // Get mesh min/max  dimensions and viewpoint
  const std::vector<double> xpos = mesh_min_max(mesh);

  // Get MPI details
  const std::size_t rank = dolfin::MPI::rank(mesh.mpi_comm());

  // Create pugi xml document
  pugi::xml_document xml_doc;

  // Write XML header
  if (MPI::rank(mesh.mpi_comm()) == 0)
    add_xml_header(xml_doc, xpos, facet_type);

  // Make a set of the indices we wish to use. In 3D, we are ignoring
  // all interior facets, so reducing the number of vertices
  // substantially
  const std::vector<std::size_t> vecindex = vertex_index(mesh);

  // Write vertices
  add_mesh(xml_doc, mesh, vecindex, facet_type);

  // Iterate over mesh facets
  std::vector<unsigned int> local_output;
  for (FaceIterator f(mesh); !f.end(); ++f)
  {
    // Check if topological dimension is 2, or if we have a boundary
    // facet in 3D
    if (tdim == 2 || f->num_global_entities(tdim) == 1)
    {
      // Get cell connected to facet
      CellIterator cell(*f);

      // Write mesh function value to string stream
      local_output.push_back((unsigned int)
                             ((meshfunction[*cell] - minval)*dval));
    }
  }

  // Gather up data on zero
  std::vector<unsigned int> gathered_output;
  MPI::gather(mesh.mpi_comm(), local_output, gathered_output);

  // Export string on root process
  if (rank == 0)
  {
    pugi::xml_node indexed_face_set = xml_doc.child("X3D").child("Scene").child("Shape").child(facet_type_to_x3d_str(facet_type));
    indexed_face_set.append_attribute("colorPerVertex") = "false";

    std::stringstream str_output;
    for (std::vector<unsigned int>::iterator val = gathered_output.begin();
         val != gathered_output.end(); ++val)
    {
      str_output << *val << " ";
    }

    indexed_face_set.append_attribute("colorIndex") = str_output.str().c_str();

    // Output palette
    pugi::xml_node color = indexed_face_set.append_child("Color");
    color.append_attribute("color") = color_palette(palette).c_str();
    // xml_doc.save_file(_filename.c_str(), "  ");
  }
  // Output string
  std::stringstream ss;
  xml_doc.save(ss, "  ");
  return ss.str();
}
*/
//-----------------------------------------------------------------------------
/*
std::string X3DOM::str(const Function& u,
                       const std::string facet_type, const size_t palette)
{
  dolfin_assert(u.function_space()->mesh());
  const Mesh& mesh = *u.function_space()->mesh();

  // Mesh geometric and topological dimensions
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
  const bool vertex_data = (dofmap.max_element_dofs() != 1);
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
      for (std::size_t j = 0; j < u.value_size() ; j++)
        val += pow(data_values[i + j*num_vertices], 2.0);
      val = sqrt(val);
      magnitude.push_back(val);
    }
    data_values.resize(magnitude.size());
    std::copy(magnitude.begin(), magnitude.end(), data_values.begin());
  }

  // Create pugi document
  pugi::xml_document xml_doc;

  // Get mesh mix/max dimensions and write XML header
  const std::vector<double> xpos = mesh_min_max(mesh);
  add_xml_header(xml_doc, xpos, facet_type);

  // Get indices of vertices on mesh surface
  const std::set<int> surface_vertices = vertex_index(mesh);
  const std::vector<std::size_t> surface_vertices_vec(surface_vertices.begin(),
                                                      surface_vertices.end());

  // Write vertices and vertex data to XML file
  add_mesh(xml_doc, mesh, surface_vertices_vec, facet_type);
  add_values_to_xml(xml_doc, mesh, surface_vertices_vec,
                    data_values, facet_type, palette);

  // Output string
  std::stringstream ss;
  if (MPI::rank(mesh.mpi_comm()) == 0)
    xml_doc.save(ss, "  ");
  return ss.str();
}
*/
//-----------------------------------------------------------------------------
/*
std::string X3DOM::html_str(const MeshFunction<std::size_t>& meshfunction,
                            const std::string facet_type, const size_t palette)
{
  // Return html string for HTML
  std::string start_str = "<html> \n"
                          "    <head> \n"
                          "        <script type='text/javascript' src='http://www.x3dom.org/download/x3dom.js'> </script> \n"
                          "        <link rel='stylesheet' type='text/css' href='http://www.x3dom.org/download/x3dom.css'></link> \n"
                          "    </head> \n"
                          "</html> \n"
                          "\n"
                          "<body>\n";

  std::stringstream ss;
  ss << start_str << xml_str(meshfunction, facet_type, palette) << "</body>";

  return ss.str();
}
*/
//-----------------------------------------------------------------------------
/*
void X3DOM::add_values_to_xml(pugi::xml_node& xml_doc, const Mesh& mesh,
                              const std::vector<std::size_t>& vecindex,
                              const std::vector<double>& data_values,
                              Representation facet_type, const std::size_t palette)
{
  const std::size_t tdim = mesh.topology().dim();

  double minval = *std::min_element(data_values.begin(), data_values.end());
  minval = MPI::min(mesh.mpi_comm(), minval);
  double maxval = *std::max_element(data_values.begin(), data_values.end());
  maxval = MPI::max(mesh.mpi_comm(), maxval);

  double scale = 0.0;
  if (maxval == minval)
    scale = 1.0;
  else
    scale = 255.0/(maxval - minval);

  std::vector<int> local_output;
  if (facet_type == Representation::Wireframe)
  {
    for (EdgeIterator e(mesh); !e.end(); ++e)
    {
      bool allow_edge = (tdim == 2);

      // If one of the faces connected to this edge is external, then
      // output the edge
      if (!allow_edge)
      {
        for (FaceIterator f(*e); !f.end(); ++f)
        {
          if (f->num_global_entities(tdim) == 1)
            allow_edge = true;
        }
      }

      if (allow_edge)
      {
        for (VertexIterator v(*e); !v.end(); ++v)
        {
          local_output.push_back((int)((data_values[v->index()] - minval)
                                       *scale));
        }
        local_output.push_back(-1);
      }
    }
  }
  else if (facet_type == Representation::Surface)
  {
    // Output faces
    for (FaceIterator f(mesh); !f.end(); ++f)
    {
      if (tdim == 2 || f->num_global_entities(tdim) == 1)
      {
        for (VertexIterator v(*f); !v.end(); ++v)
        {
          local_output.push_back((int)((data_values[v->index()] - minval)
                                       *scale));
        }
        local_output.push_back(-1);
      }
    }
  }

  // Gather up on zero
  std::vector<int> gathered_output;
  dolfin::MPI::gather(mesh.mpi_comm(), local_output, gathered_output);
  if (dolfin::MPI::rank(mesh.mpi_comm()) == 0)
  {
    // FIXME: Break this line up
    pugi::xml_node indexed_face_set
      = xml_doc.child("X3D").child("Scene").child("Shape").child(facet_type_to_x3d_str(facet_type).c_str());
    indexed_face_set.append_attribute("colorPerVertex") = "true";

    std::stringstream str_output;
    for (auto val : gathered_output)
      str_output << val << " ";
    indexed_face_set.append_attribute("colorIndex") = str_output.str().c_str();

    // Output colour palette
    pugi::xml_node color = indexed_face_set.append_child("Color");
    color.append_attribute("color") = color_palette(palette).c_str();
  }
}
*/
//-----------------------------------------------------------------------------
void X3DOM::add_mesh(pugi::xml_node& xml_node, const Mesh& mesh,
                     Representation facet_type)
{
  // Get vertex indices
  const std::set<int> vertex_indices = surface_vertex_indices(mesh);

  std::size_t offset = dolfin::MPI::global_offset(mesh.mpi_comm(),
                                                  vertex_indices.size(), true);
  const std::size_t rank = dolfin::MPI::rank(mesh.mpi_comm());
  const std::size_t tdim = mesh.topology().dim();
  const std::size_t gdim = mesh.geometry().dim();

  // Collect up topology of the local part of the mesh which should be
  // displayed
  std::vector<int> local_output;
  if (facet_type == Representation::Wireframe)
  {
    for (EdgeIterator e(mesh); !e.end(); ++e)
    {
      bool allow_edge = (tdim == 2);

      // If one of the faces connected to this edge is external, then
      // output the edge
      if (!allow_edge)
      {
        for (FaceIterator f(*e); !f.end(); ++f)
        {
          if (f->num_global_entities(tdim) == 1)
            allow_edge = true;
        }
      }

      if (allow_edge)
      {
        for (VertexIterator v(*e); !v.end(); ++v)
        {
          // Find position of vertex in set
          std::size_t pos = std::distance(vertex_indices.begin(), vertex_indices.find(v->index()));
          local_output.push_back(pos + offset);
        }
        local_output.push_back(-1);
      }
    }
  }
  else if (facet_type == Representation::Surface)
  {
    // Output faces
    for (FaceIterator f(mesh); !f.end(); ++f)
    {
      if (tdim == 2 || f->num_global_entities(tdim) == 1)
      {
        for (VertexIterator v(*f); !v.end(); ++v)
        {
          // Find position of vertex in set
          std::size_t pos = std::distance(vertex_indices.begin(), vertex_indices.find(v->index()));
          local_output.push_back(pos + offset);
        }
        local_output.push_back(-1);
      }
    }
  }

  // Gather up all topology on process 0 and append to xml
  std::vector<int> gathered_output;
  dolfin::MPI::gather(mesh.mpi_comm(), local_output, gathered_output);

  // Add topology data to XML tree
  pugi::xml_node indexed_face_set;
  if (rank == 0)
  {
    // Add edges node
    indexed_face_set = xml_node.append_child(facet_type_to_x3d_str(facet_type).c_str());
    indexed_face_set.append_attribute("solid") = "false";

    // Add data to edges node
    std::stringstream str_output;
    for (auto val : gathered_output)
      str_output << val << " ";
    indexed_face_set.append_attribute("coordIndex") = str_output.str().c_str();
  }

  // Collect up geometry of all local points
  std::vector<double> local_geom_output;
  for (auto index : vertex_indices)
  {
    Vertex v(mesh, index);
    local_geom_output.push_back(v.x(0));
    local_geom_output.push_back(v.x(1));
    if (gdim == 2)
      local_geom_output.push_back(0.0);
    else
      local_geom_output.push_back(v.x(2));
  }

  // Gather up all geometry on process 0 and append to xml
  std::vector<double> gathered_geom_output;
  dolfin::MPI::gather(mesh.mpi_comm(), local_geom_output, gathered_geom_output);
  if (rank == 0)
  {
    // Add coordinate node
    pugi::xml_node coordinate = indexed_face_set.append_child("Coordinate");

    // Add data to coordinate node
    std::stringstream str_output;
    for (auto val : gathered_geom_output)
      str_output << val << " ";
    coordinate.append_attribute("point") = str_output.str().c_str();
  }
}
//-----------------------------------------------------------------------------
pugi::xml_node X3DOM::add_xml_header(pugi::xml_node& x3d_node,
                                     const std::vector<double>& xpos,
                                     Representation facet_type,
                                     Viewpoints viewpoint_switch)
{
  pugi::xml_node scene = x3d_node.append_child("Scene");

  if (facet_type == Representation::SurfaceWithEdges)
  {
    // Append edge mesh first
    // So the facet will be on top after being appended later
    add_shape_node(scene, Representation::Wireframe);
    add_shape_node(scene, Representation::Surface);
  }
  else
  {
    add_shape_node(scene, facet_type);
  }

  // Have to append Background after shape
  pugi::xml_node background = scene.append_child("Background");
  background.append_attribute("skyColor") = "0.319997 0.340002 0.429999";

  // Append viewpoint after shape
  add_viewpoint_xml_nodes(scene, xpos, viewpoint_switch);

  // Append ambient light
  pugi::xml_node ambient_light = scene.append_child("DirectionalLight");
  ambient_light.append_attribute("ambientIntensity") = "1";
  ambient_light.append_attribute("intensity") = "0";

  return scene;
}
//-----------------------------------------------------------------------------
void X3DOM::add_viewpoint_xml_nodes(pugi::xml_node& xml_scene, 
          const std::vector<double>& xpos, Viewpoints viewpoint_switch)
{
  // FIXME: make it even shorter
  // This is center of rotation
  std::string center_of_rotation = boost::lexical_cast<std::string>(xpos[0]) + " "
    + boost::lexical_cast<std::string>(xpos[1]) + " "
    + boost::lexical_cast<std::string>(xpos[2]);

  if (viewpoint_switch == Viewpoints::On)
  {
    // Top viewpoint
    generate_viewpoint_nodes(xml_scene, 0, center_of_rotation, xpos);
    // Bottom viewpoint
    generate_viewpoint_nodes(xml_scene, 1, center_of_rotation, xpos);
    // Left viewpoint
    generate_viewpoint_nodes(xml_scene, 2, center_of_rotation, xpos);
    // Right viewpoint
    generate_viewpoint_nodes(xml_scene, 3, center_of_rotation, xpos);
    // Back viewpoint
    generate_viewpoint_nodes(xml_scene, 4, center_of_rotation, xpos);  
    // Front viewpoint
    generate_viewpoint_nodes(xml_scene, 5, center_of_rotation, xpos);
  }
  else // Just generate the front view
  {
    // Front viewpoint
    generate_viewpoint_nodes(xml_scene, 5, center_of_rotation, xpos);      
  }
}
//-----------------------------------------------------------------------------
void X3DOM::generate_viewpoint_nodes(pugi::xml_node& xml_scene, const size_t viewpoint,
					 const std::string& center_of_rotation, const std::vector<double>& xpos)
{
  std::string vp_str; // viewpoint string
  std::string ori; // orientation
  std::string pos; // position

  // Get position from string
  switch (viewpoint)
  {
	case 0: // top
	  vp_str = "Top";
	  ori = "-1 0 0 1.5707963267948";
	  pos = boost::lexical_cast<std::string>(xpos[0]) + " "
    	+ boost::lexical_cast<std::string>(xpos[1]+xpos[3]-xpos[2]) + " "
    	+ boost::lexical_cast<std::string>(xpos[2]);
	  break;
	case 1: // bottom
	  vp_str = "Bottom";
	  ori = "1 0 0 1.5707963267948";
	  pos = boost::lexical_cast<std::string>(xpos[0]) + " "
    	+ boost::lexical_cast<std::string>(xpos[1]-xpos[3]+xpos[2]) + " "
    	+ boost::lexical_cast<std::string>(xpos[2]);
	  break;  	
	case 2: // left
	  vp_str = "Left";
	  ori = "0 1 0 1.5707963267948";
	  pos = boost::lexical_cast<std::string>(xpos[0]+xpos[3]-xpos[2]) + " "
    	+ boost::lexical_cast<std::string>(xpos[1]) + " "
    	+ boost::lexical_cast<std::string>(xpos[2]);;
	  break;  	
	case 3: // right
	  vp_str = "Right";
	  ori = "0 -1 0 1.5707963267948";
	  pos = boost::lexical_cast<std::string>(xpos[0]-xpos[3]+xpos[2]) + " "
    	+ boost::lexical_cast<std::string>(xpos[1]) + " "
    	+ boost::lexical_cast<std::string>(xpos[2]);
	  break;  	
	case 4: // back
	  vp_str = "Back";
	  ori = "0 1 0 3.1415926535898";
	  pos = boost::lexical_cast<std::string>(xpos[0]) + " "
    	+ boost::lexical_cast<std::string>(xpos[1]) + " "
    	+ boost::lexical_cast<std::string>(xpos[2]-xpos[3]);
	  break;  	
	case 5: // front
	  vp_str = "Front";
	  ori = "0 0 0 1";
	  pos = boost::lexical_cast<std::string>(xpos[0]) + " "
    	+ boost::lexical_cast<std::string>(xpos[1]) + " "
    	+ boost::lexical_cast<std::string>(xpos[3]);
	  break;
	default:
	  break;
  }

  // Now get the node
  pugi::xml_node viewpoint_node = xml_scene.append_child("Viewpoint");
  viewpoint_node.append_attribute("id") = vp_str.c_str();
  viewpoint_node.append_attribute("position") = pos.c_str();

  viewpoint_node.append_attribute("orientation") = ori.c_str();
  viewpoint_node.append_attribute("fieldOfView") = "0.785398";
  viewpoint_node.append_attribute("centerOfRotation") = center_of_rotation.c_str();

  viewpoint_node.append_attribute("zNear") = "-1";
  viewpoint_node.append_attribute("zFar") = "-1";	
}
//-----------------------------------------------------------------------------
void X3DOM::add_shape_node(pugi::xml_node& x3d_scene, Representation facet_type)
{
  // pugi::xml_node shape = x3d_node.child("Scene").prepend_child("Shape");
  pugi::xml_node shape = x3d_scene.prepend_child("Shape");
  shape.append_attribute("id") = facet_type_to_x3d_str(facet_type).c_str();
  pugi::xml_node appearance = shape.append_child("Appearance");

  pugi::xml_node material = appearance.append_child("Material");
  material.append_attribute("ambientIntensity") = "0.4";
  material.append_attribute("shininess") = "0.8";
  material.append_attribute("diffuseColor") = "0.7 0.7 0.7";
  material.append_attribute("specularColor") = "0.2 0.2 0.2";
  material.append_attribute("emmisiveColor") = "0.7 0.7 0.7";
}

//-----------------------------------------------------------------------------
void X3DOM::add_doctype(pugi::xml_node& xml_node)
{
  dolfin_assert(xml_node);
  pugi::xml_node doc_type = xml_node.prepend_child(pugi::node_doctype);
  doc_type.set_value("X3D PUBLIC \"ISO//Web3D//DTD X3D 3.2//EN\" \"http://www.web3d.org/specifications/x3d-3.2.dtd\"");
}
//-----------------------------------------------------------------------------
pugi::xml_node X3DOM::add_x3d(pugi::xml_node& xml_node)
{
  pugi::xml_node x3d = xml_node.append_child("X3D");
  dolfin_assert(x3d);

  // Add on option to show rendering
  x3d.append_attribute("showStat") = "true";

  x3d.append_attribute("profile") = "Interchange";
  x3d.append_attribute("version") = "3.2";
  x3d.append_attribute("xmlns:xsd")
    = "http://www.w3.org/2001/XMLSchema-instance";
  x3d.append_attribute("xsd:noNamespaceSchemaLocation")
    = "http://www.web3d.org/specifications/x3d-3.2.xsd";
  x3d.append_attribute("width") = "500px";
  x3d.append_attribute("height") = "400px";

  return x3d;
}
//-----------------------------------------------------------------------------
void X3DOM::x3dom_xml(pugi::xml_node& xml_node, const Mesh& mesh,
                      Representation facet_type, Viewpoints viewpoint_switch)
{
  // Check that mesh is embedded in 2D or 3D
  const std::size_t gdim = mesh.geometry().dim();
  if (gdim !=2 and gdim !=3)
  {
    dolfin_error("X3DOM.cpp",
                 "get X3DOM string representation of a mesh",
                 "X3D works only for 2D and 3D meshes");
  }

  // Intialise facet-to-cell connectivity
  const std::size_t tdim = mesh.geometry().dim();
  mesh.init(tdim - 1 , tdim);

  // Get mesh max and min dimensions, needed to calculate field of
  // view
  const std::vector<double> xpos = mesh_min_max(mesh);

  // X3D doctype
  add_doctype(xml_node);

  // Add X3D node
  pugi::xml_node x3d_node = add_x3d(xml_node);
  dolfin_assert(x3d_node);

  // Add boilerplate XML no X3D node, adjusting field of view to the
  // size of the object, given by xpos
  pugi::xml_node scene = add_xml_header(x3d_node, xpos, facet_type, viewpoint_switch);
  dolfin_assert(scene);

  // FIXME: Should this go inside add_mesh?
  // Compute set of vertices that lie on boundary
  // const std::set<int> surface_vertices = surface_vertex_indices(mesh);

  // Add mesh to 'shape' XML node, based on shape id
  // First case is polygon
  if (facet_type==Representation::SurfaceWithEdges)
  {
    // First add the facet
    pugi::xml_node shape = scene.find_child_by_attribute("Shape", "id", facet_type_to_x3d_str(Representation::Surface).c_str());
    add_mesh(shape, mesh, Representation::Surface);
    // Then the edge
    shape = scene.find_child_by_attribute("Shape", "id", facet_type_to_x3d_str(Representation::Wireframe).c_str());
    add_mesh(shape, mesh, Representation::Wireframe);
  }
  else
  {
    pugi::xml_node shape = scene.find_child_by_attribute("Shape", "id", facet_type_to_x3d_str(facet_type).c_str());
    add_mesh(shape, mesh, facet_type);
  }

  // FIXME: Need to first check that node exists before accessing
  // FIXME: Really want appropiate node handle to be available, rather
  //        than having to extract it
  // Append text for mesh info
  pugi::xml_node mesh_info = x3d_node.append_child("div");
  dolfin_assert(mesh_info);
  mesh_info.append_attribute("style") = "position: absolute; bottom: 2%; left: 2%; text-align: left; font-size: 12px; color: white;";
  std::string data = "Number of vertices: " + std::to_string(mesh.num_vertices())
    + ", number of cells: " + std::to_string(mesh.num_cells());
  mesh_info.append_child(pugi::node_pcdata).set_value(data.c_str());
}
//-----------------------------------------------------------------------------
std::vector<double> X3DOM::mesh_min_max(const Mesh& mesh)
{
  // Get MPI communicator
  const MPI_Comm mpi_comm = mesh.mpi_comm();

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

  xmin = dolfin::MPI::min(mpi_comm, xmin);
  ymin = dolfin::MPI::min(mpi_comm, ymin);
  zmin = dolfin::MPI::min(mpi_comm, zmin);

  xmax = dolfin::MPI::max(mpi_comm, xmax);
  ymax = dolfin::MPI::max(mpi_comm, ymax);
  zmax = dolfin::MPI::max(mpi_comm, zmax);

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
std::set<int> X3DOM::surface_vertex_indices(const Mesh& mesh)
{
  // Get mesh toplogical dimension
  const std::size_t tdim = mesh.topology().dim();

  // Initialise connectivities
  mesh.init(tdim - 1);
  mesh.init(tdim - 1, 0);

  // Fill set of surface vertex indices
  std::set<int> vertex_set;
  for (FaceIterator f(mesh); !f.end(); ++f)
  {
    // If in 3D, only output exterior faces
    // FIXME: num_global_entities not working in serial
    if (tdim == 2 || f->num_global_entities(tdim) == 1)
    {
      for (VertexIterator v(*f); !v.end(); ++v)
        vertex_set.insert(v->index());
    }
  }

  return vertex_set;
}
//-----------------------------------------------------------------------------
std::string X3DOM::color_palette(const size_t palette)
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
      const double b
        = 1.8*exp(-pow((lm - 450.0)/((lm>450.0) ? 40.0 : 20.0), 2.0));
      const double g
        = 0.9*exp(-pow((lm - 550.0)/((lm>550.0) ? 60 : 40.0), 2.0));
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
std::string X3DOM::facet_type_to_x3d_str(Representation facet_type)
{
  // Map from enum to X3D string
  switch (facet_type)
  {
  case Representation::Surface:
    return "IndexedFaceSet";
  case Representation::Wireframe:
    return "IndexedLineSet";
  default:
    dolfin_error("X3DOM.cpp",
                 "mesh style",
                 "Unknown mesh output type");
    return "error";
  }
}
//-----------------------------------------------------------------------------

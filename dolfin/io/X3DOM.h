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

#ifndef __DOLFIN_X3DOM_H
#define __DOLFIN_X3DOM_H

#include <set>
#include <string>
#include <vector>
#include "pugixml.hpp"

namespace dolfin
{

  /// This class implements output of meshes to X3DOM XML or HTML or
  /// string

  class X3DOM
  {
  public:
  	// This simple thing doesn't work..??
  	// static std::string get_array(std::vector<double> myvec);

  	// Default value for material colour control
  	// "B3B3B3", "B3B3B3", "333333", 0.4, 0.8, 0.0
  	// Default value for background colour control
  	// "000000"


    // X3DOM representation type: facet for solid facets, and
    // edge for edges
    enum class Representation {Surface, SurfaceWithEdges, Wireframe};

    // Fixed viewpoint options
    enum class Viewpoints {On, Off};

    /// Return X3D string for a Mesh, default colour and viewpoints
    static std::string str(const Mesh& mesh, Representation facet_type);

    /// Return X3D string for a Mesh, default colour
    static std::string str(const Mesh& mesh, Representation facet_type,
    				Viewpoints viewpoint_switch);

	/// Return X3D string for a Mesh, default viewpoints
    static std::string str(const Mesh& mesh, Representation facet_type,
					const std::string& diffusive_colour,
					const std::string& emissive_colour,
					const std::string& specular_colour,
					const double ambient_intensity,
					const double shininess,
					const double transparency,
					const std::string& background_colour);    				

    /// Return X3D string for a Mesh
    static std::string str(const Mesh& mesh, Representation facet_type,
        Viewpoints viewpoint_switch, const std::string& diffusive_colour,
        							 const std::string& emissive_colour,
        							 const std::string& specular_colour,
        							 const double ambient_intensity,
        							 const double shininess,
        							 const double transparency,
        							 const std::string& background_colour);

    /// Return HTML string with embedded X3D for a Mesh
    // If Viewpoint not declared view all
    static std::string html(const Mesh& mesh, Representation facet_type);

    // Add option of Viewpoint
    static std::string html(const Mesh& mesh, Representation facet_type, 
                            Viewpoints viewpoint_switch);

    // Add option for material colour control
    // Rule to contain the material string is 
    // Diffusive Colour[3], Emissive Colour[3], Specular Colour[3]
    // Ambient Intensity [1], Shininess [1], Transparency [1]
    static std::string html(const Mesh& mesh, Representation facet_type, 
					const std::string& diffusive_colour,
					const std::string& emissive_colour,
					const std::string& specular_colour,
					const double ambient_intensity,
					const double shininess,
					const double transparency,
					const std::string& background_colour);

	static std::string html(const Mesh& mesh, Representation facet_type, 
        Viewpoints viewpoint_switch, const std::string& diffusive_colour,
        							 const std::string& emissive_colour,
        							 const std::string& specular_colour,
        							 const double ambient_intensity,
        							 const double shininess,
        							 const double transparency,
        							 const std::string& background_colour);

    // FIXME: Add option for Material Colour?
    // static std::string html(const Mesh& mesh, Representation facet_type, 
    //                         Viewpoints viewpoint_switch, Diffusive);

    // MeshFunction<std::size_t>
    //static std::string str(const MeshFunction<std::size_t>& meshfunction, const
    //                       std::string facet_type, const size_t palette);

    // Function to X3D string
    // static std::string str(const Function& function, const
    //                       std::string facet_type, const size_t palette);

    //static std::string html_str(const MeshFunction<std::size_t>& meshfunction,
    //                            const std::string facet_type, const size_t palette);

  private:
  	// Return RGB colour from hex string
  	static std::vector<double> hex2rgb(const std::string& hex);

  	// Return vector from input materials
	static std::vector<double> get_material_vector(const std::string& diffusive_colour,
	  const std::string& emissive_colour, const std::string& specular_colour,
	  const double ambient_intensity, const double shininess, const double transparency);

  	// Check the colour vectors
  	static bool check_colour(const std::vector<double>& material_colour, const std::vector<double>& bg);

    // Add X3D doctype (an XML document should have no more than one
    // doc_type node)
    static void add_doctype(pugi::xml_node& xml_node);

    // Add X3D node and attributes, and return handle to node
    static pugi::xml_node add_x3d(pugi::xml_node& xml_node);

    // Add X3DOM mesh data to XML node
    static void x3dom_xml(pugi::xml_node& xml_node, const Mesh& mesh,
      Representation facet_type, Viewpoints viewpoint_switch, 
      const std::vector<double>& material_colour, const std::vector<double>& bg);

    // Get mesh dimensions and viewpoint distance
    static std::vector<double> mesh_min_max(const Mesh& mesh);

    // Get list of vertex indices which are on surface
    static std::set<int> surface_vertex_indices(const Mesh& mesh);

    // Add mesh topology and geometry to XML, including either Facets
    // or Edges (depending on the facet_type flag). In 3D, only
    // include surface Facets/Edges.
    static void add_mesh(pugi::xml_node& xml_node, const Mesh& mesh,
                      Representation facet_type);

    // Add header to XML document, adjusting field of view to the size of the object
    static pugi::xml_node add_xml_header(pugi::xml_node& xml_node,
                                         const std::vector<double>& xpos,
                                         Representation facet_type, 
                                         Viewpoints viewpoint_switch,
                                         const std::vector<double>& material_colour,
                                         const std::vector<double>& bg);

   	// Add control tags options for html
	static void add_viewpoint_control_option(pugi::xml_node& viewpoint_control, std::string viewpoint);
    
    // Add viewpoints to scene node
    static void add_viewpoint_xml_nodes(pugi::xml_node& xml_scene, 
            const std::vector<double>& xpos, Viewpoints viewpoint_switch);

	// Generate viewpoint nodes
    static void generate_viewpoint_nodes(pugi::xml_node& xml_scene, const size_t viewpoint,
					 const std::string& center_of_rotation, const std::vector<double>& xpos);

    // Add shape node to XML document, and push the shape node to first child
    static void add_shape_node(pugi::xml_node& x3d_scene, Representation facet_type,
    							const std::vector<double>& mat_col);

    // Get a string representing a color palette (pal may be 0, 1 or 2)
    static std::string color_palette(const size_t pal);

    // Generate X3D string from facet_type
    static std::string facet_type_to_x3d_str(Representation facet_type);

	// Output values associated with Mesh points to XML using a colour
    // palette
    /*
    static void add_values_to_xml(pugi::xml_node& xml_doc, const Mesh& mesh,
                                  const std::vector<std::size_t>& vecindex,
                                  const std::vector<double>& data_values,
                                  Representation facet_type, const std::size_t palette);
    */

  };

}

#endif

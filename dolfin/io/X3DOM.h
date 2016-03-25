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

    // X3DOM representation type: facet for solid facets, and
    // wireframe for edges
    enum class FacetType {facet, wireframe};

    /// Return X3D string for a Mesh
    static std::string str(const Mesh& mesh, FacetType facet_type);

    /// Return HTML string with embedded X3D for a Mesh
    static std::string html(const Mesh& mesh, FacetType facet_type);

    // MeshFunction<std::size_t>
    //static std::string str(const MeshFunction<std::size_t>& meshfunction, const
    //                       std::string facet_type, const size_t palette);

    // Function to X3D string
    //static std::string str(const Function& function, const
    //                       std::string facet_type, const size_t palette);

    //static std::string html_str(const MeshFunction<std::size_t>& meshfunction,
    //                            const std::string facet_type, const size_t palette);

  private:

    // Add X3D doctype (an XML document should have no more than one
    // doc_type node)
    static void add_doctype(pugi::xml_node& xml_node);

    // Add X3D node and attributes, and return handle to node
    static pugi::xml_node add_x3d(pugi::xml_node& xml_node);

    // Add X3DOM mesh data to XML node
    static void x3dom_xml(pugi::xml_node& xml_node, const Mesh& mesh,
                          FacetType facet_type);

    // Get mesh dimensions and viewpoint distance
    static std::vector<double> mesh_min_max(const Mesh& mesh);

    // Get list of vertex indices which are on surface
    static std::set<int> surface_vertex_indices(const Mesh& mesh);

    // Add mesh topology and geometry to XML, including either Facets
    // or Edges (depending on the facet_type flag). In 3D, only
    // include surface Facets/Edges.
    static void add_mesh(pugi::xml_node& xml_node, const Mesh& mesh,
                         const std::set<int>& vertex_indices,
                         FacetType facet_type);

    // Output values associated with Mesh points to XML using a colour
    // palette
    /*
    static void add_values_to_xml(pugi::xml_node& xml_doc, const Mesh& mesh,
                                  const std::vector<std::size_t>& vecindex,
                                  const std::vector<double>& data_values,
                                  FacetType facet_type, const std::size_t palette);
    */

    // Add header to XML document, adjusting field of view to the size of the object
    static pugi::xml_node add_xml_header(pugi::xml_node& xml_node,
                                         const std::vector<double>& xpos,
                                         FacetType facet_type);

    // Get a string representing a color palette (pal may be 0, 1 or 2)
    static std::string color_palette(const size_t pal);

    // Generate X3D string from facet_type
    static std::string facet_type_to_x3d_str(FacetType facet_type);

  };

}

#endif

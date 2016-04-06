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

#include <array>
#include <set>
#include <string>
#include <vector>
#include "pugixml.hpp"

namespace dolfin
{

  /// Class data to store X3DOM view parameters.
  struct X3DParameters
  {
    // Developer note: X3DParameters is declared outside the X3DOM
    // class because SWIG cannot wrap nested classes.

    /// X3DOM representation type
    enum class Representation {surface, surface_with_edges, wireframe};

    /// Constructor (with default parameter settings)
    X3DParameters()
      : _representation(Representation::surface_with_edges),
        show_viewpoint_buttons(true),
        _diffuse_colour({0.1, 0.1, 0.6}),
        _emissive_colour({0.7, 0.7, 0.7}),
        _specular_colour({0.0, 0.0, 0.0}),
        _background_colour({0.95, 0.95, 0.95}),
        _ambient_intensity(1.0),
        _shininess(0.5),
        _transparency(0.0)
    {
      // Do nothing
    }

    void set_representation(Representation representation)
    { _representation = representation; }

    Representation get_representation() const
    { return _representation; }

    void set_diffuse_colour(std::array<double, 3> rgb)
    { _diffuse_colour = rgb; }

    std::array<double, 3> get_diffuse_colour() const
    { return _diffuse_colour; }

    void set_emissive_colour(std::array<double, 3> rgb)
    { _emissive_colour = rgb; }

    std::array<double, 3> get_emmisive_colour() const
    { return _emissive_colour; }

    void set_specular_colour(std::array<double, 3> rgb)
    { _specular_colour = rgb; }

    std::array<double, 3> get_specular_colour() const
    { return _specular_colour; }

    void set_background_colour(std::array<double, 3> rgb)
    { _background_colour = rgb; }

    std::array<double, 3> get_background_colour() const
    { return _background_colour; }

    void set_ambient_intensity(double intensity)
    { _ambient_intensity = intensity; }

    double get_ambient_intensity() const
    { return _ambient_intensity; }

    void set_shininess(double shininess)
    { _shininess = shininess; }

    double get_shininess() const
    { return _shininess; }

    void set_transparency(double transparency)
    { _transparency = transparency; }

    double get_transparency() const
    { return _transparency; }

    void set_viewpoint_buttons(bool show)
    { show_viewpoint_buttons = show; }

    bool get_viewpoint_buttons() const
    { return show_viewpoint_buttons; }

  private:

    // Surface, surface with edges or wireframe
    Representation _representation;

    // Toggle view point buttons
    bool show_viewpoint_buttons;

    // TODO: document
    // See http://doc.x3dom.org/author/Shape/Material.html
    std::array<double, 3> _diffuse_colour, _emissive_colour, _specular_colour,
      _background_colour;

    // TODO: document
    double _ambient_intensity, _shininess, _transparency;
  };


  /// This class implements output of meshes to X3DOM XML or HTML or
  /// string

  class X3DOM
  {
  public:

    /// Return X3D string for a Mesh
    static std::string str(const Mesh& mesh,
                           X3DParameters paramemeters=X3DParameters());

    /// Return HTML string with embedded X3D
    static std::string html(const Mesh& mesh,
                            X3DParameters parameters=X3DParameters());

  private:

    // FIXME: This should be a C++11 style enum (enum Viewpoint class
    // {...};), but a Swig bug needa fixing
    // (https://github.com/swig/swig/issues/594)
    enum  Viewpoint {top, bottom, left, right, back, front, default_view};

    // Add X3D doctype (an XML document should have no more than one
    // doc_type node)
    static void add_doctype(pugi::xml_node& xml_node);

    // Add X3D node and attributes, and return handle to node
    static pugi::xml_node add_x3d_node(pugi::xml_node& xml_node);

    // Add X3DOM mesh data to XML node
    static void add_x3dom_data(pugi::xml_node& xml_node, const Mesh& mesh,
                               const X3DParameters& parameters);

    // Add mesh topology and geometry to XML, including either Facets
    // or Edges (depending on the facet_type flag). In 3D, only
    // include surface Facets/Edges.
    static void add_mesh_data(pugi::xml_node& xml_node, const Mesh& mesh,
                              const X3DParameters& parameters,
                              bool surface);

    // Add control tags options for html
    static void add_viewpoint_control_option(pugi::xml_node& viewpoint_control,
                                             std::string viewpoint);

    // Add viewpoints to scene node (X3D)
    static void add_viewpoint_nodes(pugi::xml_node& xml_scene,
                                    const std::vector<double>& xpos,
                                    bool show_viewpoint_buttons);

    // Add a single viewpoint node (X3D)
    static void add_viewpoint_node(pugi::xml_node& xml_scene,
                                   Viewpoint viewpoint,
                                   const std::string center_of_rotation,
                                   const std::vector<double>& xpos);

    // Get mesh dimensions and viewpoint distance
    static std::vector<double> mesh_min_max(const Mesh& mesh);

    // Get list of vertex indices which are on surface
    static std::set<int> surface_vertex_indices(const Mesh& mesh);

    // Build topology and geometry data from a Mesh ready for X3DOM
    // output
    static void build_mesh_data(std::vector<int>& topology,
                                std::vector<double>& geometry,
                                const Mesh& mesh, bool surface);

    // Return "x[0] x[1] x[2]" string from array of colour RGB
    static std::string array_to_string3(std::array<double, 3> x);

  };

}

#endif

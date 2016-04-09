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
#include <utility>
#include <vector>
#include <boost/multi_array.hpp>

#include <dolfin/geometry/Point.h>

// TODO:
// - Can we order vertices so that attribute 'solid' can be set to true?
// - Make size (height, width) a parameter
// - Add ambient intensity parameter
// - Test and add more sanity checks for Functions
// - Add support for GenericFunction
// - Add support for P0 Functions
// - Add support for MeshFunctions

namespace pugi
{
  class xml_node;
  class xml_document;
}

namespace dolfin
{

  /// Class data to store X3DOM view parameters.
  class X3DOMParameters
  {
    // Developer note: X3DOMParameters is declared outside the X3DOM
    // class because SWIG cannot wrap nested classes.

  public:

    /// X3DOM representation type
    enum class Representation {surface, surface_with_edges, wireframe};

    /// Constructor (with default parameter settings)
    X3DOMParameters();

    // Set representation of object (wireframe, surface or
    // surface_with_edges)
    void set_representation(Representation representation);

    Representation get_representation() const;

    // Sets the RGB color of the object
    void set_diffuse_color(std::array<double, 3> rgb);

    // Get diffuse color
    std::array<double, 3> get_diffuse_color() const;

    // Set the RGB colour of lines
    void set_emissive_color(std::array<double, 3> rgb);

    // Get the emissive color
    std::array<double, 3> get_emissive_color() const;

    void set_specular_color(std::array<double, 3> rgb);
    std::array<double, 3> get_specular_color() const;

    // Set background RGB color
    void set_background_color(std::array<double, 3> rgb);

    // Get background color
    std::array<double, 3> get_background_color() const;

    void set_ambient_intensity(double intensity);

    double get_ambient_intensity() const;

    void set_shininess(double shininess);

    double get_shininess() const;

    void set_transparency(double transparency);

    double get_transparency() const;

    // Toggle viewpoint buttons
    void set_viewpoint_buttons(bool show);

    bool get_viewpoint_buttons() const;

    // Turn X3D 'stats' window on
    void set_x3d_stats(bool show);
    bool get_x3d_stats() const;

  private:

    // Check that RGB colors are valid. Throws error if value is
    // not invalid.
    static void check_rgb(std::array<double, 3>& rgb);

    // Check that value is valid. Throws error if value is not
    // invalid.
    static void check_value_range(double value, double lower, double upper);

    // Surface, surface with edges or wireframe
    Representation _representation;

    // Toggle view point buttons
    bool _show_viewpoints;

    // TODO: document
    // RGB colours, see http://doc.x3dom.org/author/Shape/Material.html
    std::array<double, 3> _diffuse_color, _emissive_color, _specular_color,
      _background_color;

    // TODO: document
    double _ambient_intensity, _shininess, _transparency;

    // Turn X3D stats on/off
    bool _show_x3d_stats;
  };

  // Forward declarations
  class Function;
  class Mesh;
  class Point;

  /// This class implements output of meshes to X3DOM XML or XHTML or
  /// string. We use XHTML as it has full support for X3DOM, and fits
  /// better with the use of an XML library to produce document tree.
  ///
  /// When creating stand-along HTML files, it is necessary to use the
  /// file extension '.xhtml'.

  class X3DOM
  {
  public:

    /// Return X3D string for a Mesh
    static std::string str(const Mesh& mesh,
                           X3DOMParameters paramemeters=X3DOMParameters());

    /// Return XHTML string with embedded X3D for a Mesh
    static std::string xhtml(const Mesh& mesh,
                             X3DOMParameters parameters=X3DOMParameters());

    /// Return X3D string for a Function
    static std::string str(const Function& u,
                           X3DOMParameters paramemeters=X3DOMParameters());

    /// Return XHTML string with embedded X3D for a Function
    static std::string xhtml(const Function& u,
                             X3DOMParameters parameters=X3DOMParameters());

  private:

    // FIXME: This should be a C++11 style enum (enum Viewpoint class
    // {...};), but a Swig bug needs fixing
    // (https://github.com/swig/swig/issues/594)
    enum  Viewpoint {top, bottom, left, right, back, front, default_view};

    // Build X3DOM pugixml tree
    static void x3dom(pugi::xml_document& xml_doc, const Mesh& mesh,
                      const std::vector<double>& vertex_values,
                      const std::vector<double>& facet_values,
                      const X3DOMParameters& parameters);

    // Build XHTML pugixml tree
    static void xhtml(pugi::xml_document& xml_doc, const Mesh& mesh,
                      const std::vector<double>& vertex_values,
                      const std::vector<double>& facet_values,
                      const X3DOMParameters& parameters);

    // Add HTML preamble (HTML) to XML doc and return 'html' node
    static pugi::xml_node add_html_preamble(pugi::xml_node& xml_node);

    // Add X3D doctype (an XML document should have no more than one
    // doc_type node)
    static void add_x3dom_doctype(pugi::xml_node& xml_node);

    // Add HTML doctype (an XML document should have no more than one
    // doc_type node)
    static void add_html_doctype(pugi::xml_node& xml_node);

    // Add X3D node and attributes, and return handle to node (x3D)
    static pugi::xml_node add_x3d_node(pugi::xml_node& xml_node,
                                       std::array<double, 2> size,
                                       bool show_stats);

    // Add X3DOM Mesh data to XML node (X3D)
    static void add_x3dom_data(pugi::xml_node& xml_node, const Mesh& mesh,
                               const std::vector<double>& vertex_values,
                               const std::vector<double>& facet_values,
                               const X3DOMParameters& parameters);

    // Add mesh topology and geometry to XML. 'surface' flag controls
    // surface vs wireframe representation (X3D)
    static void add_mesh_data(pugi::xml_node& xml_node, const Mesh& mesh,
                              const std::vector<double>& vertex_values,
                              const std::vector<double>& facet_values,
                              const X3DOMParameters& parameters,
                              bool surface);

    // Add viewpoint control (HTML)
    static void add_viewpoint_control_option(pugi::xml_node& viewpoint_control,
                                             std::string viewpoint);

    // Add a collection viewpoint nodes (X3D)
    static void add_viewpoint_nodes(pugi::xml_node& xml_scene,
                                    const Point p, double d,
                                    bool show_viewpoint_buttons);

    // Add a single viewpoint node (X3D)
    static void add_viewpoint_node(pugi::xml_node& xml_scene,
                                   Viewpoint viewpoint,
                                   const Point p,
                                   const double s);

    // Return RGB color map (256 values)
    static boost::multi_array<float, 2> color_map();

    // Get mesh dimensions and viewpoint distance
    static std::pair<Point, double> mesh_min_max(const Mesh& mesh);

    // Get list of vertex indices which are on surface
    static std::set<int> surface_vertex_indices(const Mesh& mesh);

    // Build topology and geometry data from a Mesh ready for X3DOM
    // output
    static void build_mesh_data(std::vector<int>& topology,
                                std::vector<double>& geometry,
                                std::vector<double>& value_data,
                                const Mesh& mesh,
                                const std::vector<double>& vertex_values,
                                const std::vector<double>& facet_values,
                                bool surface);

    // Return "x[0] x[1] x[2]" string from array of color RGB
    static std::string array_to_string3(std::array<double, 3> x);

    // Utility to convert pugi::xml_document into a std::string
    static std::string to_string(pugi::xml_document& xml_doc);

  };

}

#endif

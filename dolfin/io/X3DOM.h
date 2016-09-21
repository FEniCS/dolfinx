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
// - Add ambient intensity parameter
// - Test and add more sanity checks for Functions
// - Add support for GenericFunction
// - Add support for MeshFunctions
// - Add support for DG1 Functions
// - Add vector support (arrows) - including correct placement for RT0
//   etc. - advanced
// - Document all class methods below properly

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

    /// Set representation of object (wireframe, surface or
    /// surface_with_edges)
    void set_representation(Representation representation);

    /// Get the current representation of the object (wireframe,
    /// surface or surface_with_edges)
    Representation get_representation() const;

    /// Get the size of the viewport
    std::array<double, 2> get_viewport_size() const;

    /// Set the RGB color of the object
    void set_diffuse_color(std::array<double, 3> rgb);

    /// Get the RGB diffuse color of the object
    std::array<double, 3> get_diffuse_color() const;

    /// Set the RGB emissive color
    void set_emissive_color(std::array<double, 3> rgb);

    /// Get the RGB emissive color
    std::array<double, 3> get_emissive_color() const;

    /// Set the RGB specular color
    void set_specular_color(std::array<double, 3> rgb);

    /// Get the RGB specular color
    std::array<double, 3> get_specular_color() const;

    /// Set background RGB color
    void set_background_color(std::array<double, 3> rgb);

    /// Get background RGB color
    std::array<double, 3> get_background_color() const;

    /// Set the ambient lighting intensity
    void set_ambient_intensity(double intensity);

    /// Get the ambient lighting intensity
    double get_ambient_intensity() const;

    /// Set the surface shininess of the object
    void set_shininess(double shininess);

    /// Set the surface shininess of the object
    double get_shininess() const;

    /// Set the transparency (0-1)
    void set_transparency(double transparency);

    /// Get the transparency (0-1)
    double get_transparency() const;

    /// Set the color map by supplying a vector of 768 values
    /// (256*RGB) (using std::vector for Python compatibility via
    /// SWIG)
    void set_color_map(const std::vector<double>& color_data);

    /// Get the color map as a vector of 768 values (256*RGB) (using
    /// std::vector for Python compatibility via SWIG)
    std::vector<double> get_color_map() const;

    /// Get the color map as a boost::multi_array (256x3)
    boost::multi_array<float, 2> get_color_map_array() const;

    /// Turn X3D 'statistics' window on/off
    void set_x3d_stats(bool show);

    /// Get the state of the 'statistics' window
    bool get_x3d_stats() const;

    /// Toggle menu option
    void set_menu_display(bool show);

    /// Get the menu display state
    bool get_menu_display() const;

  private:

    // Check that RGB colors are valid. Throws error if value is not
    // invalid.
    static void check_rgb(std::array<double, 3>& rgb);

    // Check that value is valid. Throws error if value is not
    // invalid.
    static void check_value_range(double value, double lower, double upper);

    // Return a default RGB color map (256 values)
    static boost::multi_array<float, 2> default_color_map();

    // Surface, surface with edges or wireframe
    Representation _representation;

    // Dimensions of viewing area
    std::array<double, 2> _size;

    // TODO: document
    // RGB colours, see http://doc.x3dom.org/author/Shape/Material.html
    std::array<double, 3> _diffuse_color, _emissive_color, _specular_color,
      _background_color;

    // TODO: document
    double _ambient_intensity, _shininess, _transparency;

    // RGB color map (256 values)
    boost::multi_array<float, 2> _color_map;

    // Turn X3D stats on/off
    bool _show_x3d_stats;

    // Turn menu on/off
    bool _menu_display;
  };

  // Forward declarations
  class Function;
  class Mesh;
  class Point;

  /// This class implements output of meshes to X3DOM XML or HTML5
  /// with X3DOM strings. The latter can be used for interactive
  /// visualisation
  //
  // Developer note: pugixml is used to created X3DOM and HTML5. By
  // using pugixml, we produce valid XML, but care must be taken that
  // the XML is also valid HTML. This includes not letting pugixml
  // create self-closing elements, in cases. E.g., <foo
  // bar="foobar"></foo> is fine, but the self-closing syntax <foo
  // bar="foobar" /> while being valid XML is is not valid HTML5. See
  // https://github.com/x3dom/x3dom/issues/600.
  //

  class X3DOM
  {
  public:

    /// Return X3D string for a Mesh
    static std::string str(const Mesh& mesh,
                           X3DOMParameters parameters=X3DOMParameters());

    /// Return X3D string for a Function
    static std::string str(const Function& u,
                           X3DOMParameters parameters=X3DOMParameters());

    /// Return HTML5 string with embedded X3D for a Mesh
    static std::string html(const Mesh& mesh,
                            X3DOMParameters parameters=X3DOMParameters());


    /// Return HTML5 string with embedded X3D for a Function
    static std::string html(const Function& u,
                            X3DOMParameters parameters=X3DOMParameters());


    /// Build X3DOM pugixml tree for a Mesh
    static void
      build_x3dom_tree(pugi::xml_document& xml_doc,
                       const Mesh& mesh,
                       const X3DOMParameters& parameters=X3DOMParameters());

    /// Build X3DOM pugixml tree for a Function
    static void
      build_x3dom_tree(pugi::xml_document& xml_doc,
                       const Function& u,
                       const X3DOMParameters& parameters=X3DOMParameters());

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

    // Build HTML pugixml tree
    static void html(pugi::xml_document& xml_doc, const Mesh& mesh,
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


    // Add a collection viewpoint nodes (X3D)
    static void add_viewpoint_nodes(pugi::xml_node& xml_scene,
                                    const Point p, double d,
                                    bool show_viewpoint_buttons);

    // Add a single viewpoint node (X3D)
    static void add_viewpoint_node(pugi::xml_node& xml_scene,
                                   Viewpoint viewpoint,
                                   const Point p,
                                   const double s);

    // Add the menu display and the desired subsections
    static void add_menu_display(pugi::xml_node& xml_node, const Mesh& mesh,
                                 const X3DOMParameters& parameters);

    // Add the button for a tab to be added to the menu
    static void add_menu_tab_button(pugi::xml_node& xml_node, std::string name,
                                    bool checked);

    // Create a generic content node to hold specific menu content
    static pugi::xml_node create_menu_content_node(pugi::xml_node& xml_node,
                                                   std::string name, bool show);

    // Add the options tab in the menu display
    static void add_menu_options_tab(pugi::xml_node& xml_node);

    // Add an option to the options tab in menu display
    static void add_menu_options_option(pugi::xml_node& xml_node,
                                        std::string name);

    // Add the summary tab in the menu display
    static void add_menu_summary_tab(pugi::xml_node& xml_node,
                                     const Mesh& mesh);

    // Add the color tab in the menu display
    static void add_menu_color_tab(pugi::xml_node& xml_node);

    // Add the warp tab in the menu display
    static void add_menu_warp_tab(pugi::xml_node& xml_node);

    // Add the viewpoint tab in the menu display
    static void add_menu_viewpoint_tab(pugi::xml_node& xml_node);

    // Add a viewpoint button to the appropriate parent
    static void add_menu_viewpoint_button(pugi::xml_node& xml_node,
                                          std::string name);

    // Get centre point of mesh bounds, and a reasonable viewpoint
    // distance from it
    static std::pair<Point, double> mesh_centre_and_distance(const Mesh& mesh);

    // Get list of vertex indices which are on surface
    static std::set<int> surface_vertex_indices(const Mesh& mesh);

    // Get the values of a function at vertices, (or on facets for P0)
    static void get_function_values(const Function& u,
                                    std::vector<double>& vertex_values,
                                    std::vector<double>& facet_values);

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
    static std::string to_string(pugi::xml_document& xml_doc,
                                 unsigned int flags);

  };

}

#endif

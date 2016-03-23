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
// First added:  2012-03-05
// Last changed: 2013-05-10

#ifndef __X3DOM_H
#define __X3DOM_H

#include "pugixml.hpp"

namespace dolfin
{

  /// This class implements output of meshes to X3DOM XML 
  /// or HTML or string
  class X3DOM
  {
  public:

    /// Constructor
    X3DOM();

    /// Destructor
    ~X3DOM();

    // Whether in Face or Edge mode - should either be
    // "IndexedFaceSet" or "IndexedLineSet"
    static std::string xml_str(const Mesh& mesh, const std::string facet_type, const size_t palette);

    // static pugi::xml_document xml_doc(const Mesh& mesh);

    static std::string html_str(const Mesh& mesh, const std::string facet_type, const size_t palette);

    // static pugi::xml_document html_doc(const Mesh& mesh);

    // void xml_to_file(const std::string filename);

    // void html_to_file(const std::string filename);

  private:
    // Get mesh dimensions and viewpoint distance
    static std::vector<double> mesh_min_max(const Mesh& mesh);

    // Get list of vertex indices which are on surface
    static std::vector<std::size_t> vertex_index(const Mesh& mesh);

    // Output mesh vertices to XML
    static void write_vertices(pugi::xml_document& xml_doc, const Mesh& mesh,
                        const std::vector<std::size_t> vecindex, const std::string facet_type);

    // Output values to XML using a colour palette
    static void write_values(pugi::xml_document& xml_doc, const Mesh& mesh,
                      const std::vector<std::size_t> vecindex,
                      const std::vector<double> data_values, const std::string facet_type, const std::size_t palette);

    // XML header output
    static void output_xml_header(pugi::xml_document& xml_doc,
                           const std::vector<double>& xpos, const std::string facet_type);

    // Get a string representing a color palette
    static std::string color_palette(const size_t pal);

  };

}

#endif

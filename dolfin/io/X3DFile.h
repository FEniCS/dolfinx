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

#ifndef __X3D_FILE_H
#define __X3D_FILE_H

#include "GenericFile.h"

namespace pugi
{
  class xml_document;
}

namespace dolfin
{

  /// This class implements output of meshes to X3D (successor to
  /// VRML) graphics format (http://www.web3d.org/x3d/). It is
  /// suitable for output of small to medium size meshes for 3D
  /// visualisation via browsers, and can also do basic Function and
  /// MeshFunction output (on the surface) X3D files can be included
  /// on web pages with WebGL functionality (see www.x3dom.org).

  class X3DFile : public GenericFile
  {
  public:

    /// Constructor
    explicit X3DFile(const std::string filename);

    /// Destructor
    ~X3DFile();

    /// Output Mesh to X3D format - for a 3D mesh, only producing
    /// surface facets
    void operator<< (const Mesh& mesh);

    /// Output MeshFunction to X3D format - for a 3D mesh, only
    /// producing surface facets
    void operator<< (const MeshFunction<std::size_t>& meshfunction);

    /// Output Function to X3D format - for a 3D mesh, only producing
    /// surface facets
    void operator<< (const Function& function);

  private:

    // Get mesh dimensions and viewpoint distance
    std::vector<double> mesh_min_max(const Mesh& mesh) const;

    // Get list of vertex indices which are on surface
    std::vector<std::size_t> vertex_index(const Mesh& mesh) const;

    // Output mesh vertices to XML
    void write_vertices(pugi::xml_document& xml_doc, const Mesh& mesh,
                        const std::vector<std::size_t> vecindex);

    // XML header output
    void output_xml_header(pugi::xml_document& xml_doc,
                           const std::vector<double>& xpos);

    // Write out surface mesh to file
    void write_meshfunction(const MeshFunction<std::size_t>& meshfunction);

    // Get a string representing a color palette
    std::string color_palette(const int pal) const;

    // Whether in Face or Edge mode - should either be
    // "IndexedFaceSet" or "IndexedLineSet"
    const std::string facet_type;

  };

}

#endif

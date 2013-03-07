// Copyright (C) 2012 Benjamin Kehlet
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
// Modified by Joachim B Haga 2012
//
// First added:  2012-05-10
// Last changed: 2012-09-06

#ifndef __CSG_CGAL_MESH_GENERATOR3D_H
#define __CSG_CGAL_MESH_GENERATOR3D_H

#include <dolfin/common/Variable.h>
#include <boost/shared_ptr.hpp>

namespace dolfin
{

  // Forward declarations
  class Mesh;
  class CSGGeometry;

  /// Mesh generator for Constructive Solid Geometry (CSG)
  /// utilizing CGALs boolean operation on Nef_polyhedrons.

  class CSGCGALMeshGenerator3D : public Variable
  {
  public :
    CSGCGALMeshGenerator3D(const CSGGeometry& geometry);

    CSGCGALMeshGenerator3D(boost::shared_ptr<const CSGGeometry> geometry);

    ~CSGCGALMeshGenerator3D();

    void generate(Mesh& mesh) const;

    void save_off(std::string filename) const;

    /// Default parameter values
    static Parameters default_parameters()
    {
      Parameters p("csg_cgal_meshgenerator");
      p.add("mesh_resolution", 64);
      p.add("perturb_optimize", false);
      p.add("exude_optimize", false);
      p.add("lloyd_optimize", false);
      p.add("odt_optimize", false);
      p.add("edge_size", 0.025);
      p.add("facet_angle", 25.0);
      p.add("facet_size", 0.05);
      p.add("facet_distance", 0.005);
      p.add("cell_radius_edge_ratio", 3.0);
      p.add("cell_size", 0.05);
      p.add("remove_degenerated", true);
      p.add("detect_sharp_features", true);

      return p;
    }

  private:
    boost::shared_ptr<const CSGGeometry> _geometry;
  };

}

#endif

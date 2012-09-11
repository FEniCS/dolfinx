// Copyright (C) 2012 Joachim B Haga
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
// First added:  2012-08-27
// Last changed: 2012-09-11

#ifndef __VTK_PLOTTABLE_CSGGEOMETRY_H
#define __VTK_PLOTTABLE_CSGGEOMETRY_H

#ifdef HAS_VTK

#include "VTKPlottableMesh.h"

namespace dolfin
{

  class CSGGeometry;

  /// Data wrapper class for CSG geometries

  class VTKPlottableCSGGeometry : public VTKPlottableMesh
  {
  public:

    explicit
    VTKPlottableCSGGeometry(boost::shared_ptr<const CSGGeometry> geometry);

    /// Additional parameters for VTKPlottableCSGGeometry
    virtual Parameters default_parameters()
    {
      Parameters p;
      p.add("wireframe", true);
      p.add("scalarbar", false);
      return p;
    }

    /// Update the plottable data
    void update(boost::shared_ptr<const Variable> var, const Parameters& parameters, int frame_counter);

    /// Return whether this plottable is compatible with the variable
    bool is_compatible(const Variable &var) const;

  private:

    boost::shared_ptr<const CSGGeometry> _geometry;

  };

  VTKPlottableCSGGeometry *CreateVTKPlottable(boost::shared_ptr<const CSGGeometry> geometry);
}

#endif

#endif

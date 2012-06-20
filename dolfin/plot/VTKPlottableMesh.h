// Copyright (C) 2012 Fredrik Valdmanis
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
// First added:  2012-06-20
// Last changed: 2012-06-20

#ifndef __VTK_PLOTTABLE_MESH_H
#define __VTK_PLOTTABLE_MESH_H

#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkGeometryFilter.h>
#include <vtkAlgorithmOutput.h>

#include <dolfin/mesh/Mesh.h>

#ifdef HAS_VTK

namespace dolfin
{
  class VTKPlottableMesh
  {
  public:

    explicit VTKPlottableMesh(boost::shared_ptr<const Mesh> mesh);

    /// Update the VTK grid. Should be called after every update to the mesh
    void update();

    void update_range(double range[2]);

    vtkSmartPointer<vtkAlgorithmOutput> get_output() const;

  protected:

    // The VTK grid constructed from the DOLFIN mesh
    vtkSmartPointer<vtkUnstructuredGrid> _grid;

    // The geometry filter
    vtkSmartPointer<vtkGeometryFilter> _geometryFilter;

    // The mesh to visualize
    boost::shared_ptr<const Mesh> _mesh;

  private:


  };

}

#endif

#endif

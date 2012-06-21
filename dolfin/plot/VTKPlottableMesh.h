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
// Last changed: 2012-06-21

#ifndef __VTK_PLOTTABLE_MESH_H
#define __VTK_PLOTTABLE_MESH_H

#ifdef HAS_VTK

#include <vtkUnstructuredGrid.h>
#include <vtkGeometryFilter.h>

#include <dolfin/mesh/Mesh.h>

#include "GenericVTKPlottable.h"

namespace dolfin
{
  class VTKPlottableMesh : public GenericVTKPlottable
  {
  public:

    explicit VTKPlottableMesh(boost::shared_ptr<const Mesh> mesh);

    //--- Implementation of the GenericVTKPlottable interface ---

    /// Initialize the parts of the pipeline that this class controls
    void init_pipeline();

    /// Update the plottable data
    void update(const Parameters& parameters);

    /// Update the scalar range of the plottable data
    void update_range(double range[2]);

    /// Return data to visualize
    vtkSmartPointer<vtkAlgorithmOutput> get_output() const;

    /// Get an actor for showing vertex labels
    vtkSmartPointer<vtkActor2D> get_vertex_label_actor();

  protected:

    // The VTK grid constructed from the DOLFIN mesh
    vtkSmartPointer<vtkUnstructuredGrid> _grid;

    // The geometry filter
    vtkSmartPointer<vtkGeometryFilter> _geometryFilter;

    // The mesh to visualize
    boost::shared_ptr<const Mesh> _mesh;

    // The label actor
    vtkSmartPointer<vtkActor2D> _vertexLabelActor;

  private:


  };

}

#endif

#endif

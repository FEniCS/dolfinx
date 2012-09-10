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
// Modified by Joachim B Haga 2012
//
// First added:  2012-06-20
// Last changed: 2012-09-10

#ifndef __VTK_PLOTTABLE_MESH_H
#define __VTK_PLOTTABLE_MESH_H

#ifdef HAS_VTK

#include <vtkSmartPointer.h>

#include <dolfin/common/types.h>
#include <dolfin/mesh/Mesh.h>

#include "GenericVTKPlottable.h"

class vtkActor2D;
class vtkActor3D;
class vtkAlgorithmOutput;
class vtkGeometryFilter;
class vtkIdFilter;
class vtkPointSetAlgorithm;
class vtkUnstructuredGrid;

namespace dolfin
{

  class Mesh;
  class Parameters;
  class Variable;
  class VTKWindowOutputStage;

  /// Data wrapper class for plotting meshes. It also acts as a superclass
  /// for the other data wrapper classes, as all kinds of plottable data
  /// also holds a mesh.

  class VTKPlottableMesh : public GenericVTKPlottable
  {
  public:

    explicit VTKPlottableMesh(boost::shared_ptr<const Mesh> mesh, uint entity_dim);

    explicit VTKPlottableMesh(boost::shared_ptr<const Mesh> mesh);

    //--- Implementation of the GenericVTKPlottable interface ---

    /// Initialize the parts of the pipeline that this class controls
    virtual void init_pipeline(const Parameters &parameters);

    /// Connect or reconnect to the output stage.
    virtual void connect_to_output(VTKWindowOutputStage& output);

    /// Update the plottable data
    virtual void update(boost::shared_ptr<const Variable> var, const Parameters& parameters, int frame_counter);

    /// Return whether this plottable is compatible with the variable
    virtual bool is_compatible(const Variable &var) const;

    /// Update the scalar range of the plottable data
    virtual void update_range(double range[2]);

    /// Return geometric dimension
    virtual uint dim() const;

    /// Get an actor for showing vertex labels
    vtkSmartPointer<vtkActor2D> get_vertex_label_actor(vtkSmartPointer<vtkRenderer>);

    /// Get an actor for showing cell labels
    vtkSmartPointer<vtkActor2D> get_cell_label_actor(vtkSmartPointer<vtkRenderer>);

    /// Get an actor for showing the mesh
    vtkSmartPointer<vtkActor> get_mesh_actor();

  protected:

    /// Get the output port
    virtual vtkSmartPointer<vtkAlgorithmOutput> get_output() const;

    // Create label filter
    void build_id_filter();

    // Build the grid from mesh
    void build_grid_cells(vtkSmartPointer<vtkUnstructuredGrid> &grid, uint entity_dim);

    /// Set scalar values on the mesh
    template <class T>
    void setPointValues(uint size, const T *indata, const Parameters &parameters);

    /// Set scalar values on the mesh
    template <class T>
    void setCellValues(uint size, const T *indata, const Parameters &parameters);

    boost::shared_ptr<const Mesh> mesh() const;

    vtkSmartPointer<vtkUnstructuredGrid> grid() const;

    void insert_filter(vtkSmartPointer<vtkPointSetAlgorithm> filter);

  private:

    // The possibly lower-dimensional VTK grid constructed from the DOLFIN mesh
    vtkSmartPointer<vtkUnstructuredGrid> _grid;

    // The full-dimensional VTK grid constructed from the DOLFIN mesh
    vtkSmartPointer<vtkUnstructuredGrid> _full_grid;

    // The geometry filter
    vtkSmartPointer<vtkGeometryFilter> _geometryFilter;

    // The mesh to visualize
    boost::shared_ptr<const Mesh> _mesh;

    // The label actors
    vtkSmartPointer<vtkActor2D> _vertexLabelActor;
    vtkSmartPointer<vtkActor2D> _cellLabelActor;
    vtkSmartPointer<vtkIdFilter> _idFilter;

    // The mesh actor
    vtkSmartPointer<vtkActor> _meshActor;

    // The dimension of the facets
    const uint _entity_dim;

  };

  VTKPlottableMesh *CreateVTKPlottable(boost::shared_ptr<const Mesh> mesh);
}

#endif

#endif


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
// Last changed: 2012-08-21

#ifndef __VTK_PLOTTABLE_MESH_H
#define __VTK_PLOTTABLE_MESH_H

#ifdef HAS_VTK

#include <vtkUnstructuredGrid.h>
#include <vtkGeometryFilter.h>
#include <vtkFloatArray.h>
#include <vtkCellArray.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkVectorNorm.h>

#include "GenericVTKPlottable.h"

namespace dolfin
{

  class Mesh;

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
    void init_pipeline();

    /// Update the plottable data
    void update(boost::shared_ptr<const Variable> var, const Parameters& parameters, int frame_counter);

    bool is_compatible(const Variable &var) const { return dynamic_cast<const Mesh*>(&var); }

    /// Update the scalar range of the plottable data
    void update_range(double range[2]);

    /// Return geometric dimension
    virtual uint dim();

    /// Return data to visualize
    vtkSmartPointer<vtkAlgorithmOutput> get_output() const;

    /// Get an actor for showing vertex labels
    vtkSmartPointer<vtkActor2D> get_vertex_label_actor();

  protected:

    /// Set scalar values on the mesh
    template <class T>
    void setPointValues(uint size, const T* indata);

    /// Set scalar values on the mesh
    template <class T>
    void setCellValues(uint size, const T* indata);

    // The VTK grid constructed from the DOLFIN mesh
    vtkSmartPointer<vtkUnstructuredGrid> _grid;

    // The geometry filter
    vtkSmartPointer<vtkGeometryFilter> _geometryFilter;

    // The mesh to visualize
    boost::shared_ptr<const Mesh> _mesh;

    // The label actor
    vtkSmartPointer<vtkActor2D> _vertexLabelActor;

    // The dimension of the facets
    const uint _entity_dim;

  private:

    std::vector<uint> _entity_map;

  };

  //---------------------------------------------------------------------------
  // Implementation of VTKPlottableMeshFunction
  //---------------------------------------------------------------------------
  template <class T>
  void VTKPlottableMesh::setPointValues(uint size, const T* indata)
  {
    const uint num_vertices = _mesh->num_vertices();
    const uint num_components = size / num_vertices;

    dolfin_assert(num_components > 0 && num_components <= 3);
    dolfin_assert(num_vertices*num_components == size);

    vtkSmartPointer<vtkFloatArray> values =
      vtkSmartPointer<vtkFloatArray>::New();
    if (num_components == 1)
    {
      values->SetNumberOfValues(num_vertices);
      for (uint i = 0; i < num_vertices; ++i)
      {
        values->SetValue(i, (double)indata[i]);
      }
      _grid->GetPointData()->SetScalars(values);
    }
    else
    {
      // NOTE: Allocation must be done in this order!
      // Note also that the number of VTK vector components must always be 3
      // regardless of the function vector value dimension
      values->SetNumberOfComponents(3);
      values->SetNumberOfTuples(num_vertices);
      for (uint i = 0; i < num_vertices; ++i)
      {
        // The entries in "vertex_values" must be copied to "vectors". Viewing
        // these arrays as matrices, the transpose of vertex values should be copied,
        // since DOLFIN and VTK store vector function values differently
        for (uint d = 0; d < num_components; d++)
        {
          values->SetValue(3*i+d, indata[i+num_vertices*d]);
        }
        for (uint d = num_components; d < 3; d++)
        {
          values->SetValue(3*i+d, 0.0);
        }
      }
      _grid->GetPointData()->SetVectors(values);

      // Compute norms of vector data
      vtkSmartPointer<vtkVectorNorm> norms =
        vtkSmartPointer<vtkVectorNorm>::New();
      norms->SetInput(_grid);
      norms->SetAttributeModeToUsePointData();
      //NOTE: This update is necessary to actually compute the norms
      norms->Update();

      // Attach vector norms as scalar point data in the VTK grid
      _grid->GetPointData()->SetScalars(
          norms->GetOutput()->GetPointData()->GetScalars());
    }
  }
  //----------------------------------------------------------------------------
  template <class T>
  void VTKPlottableMesh::setCellValues(uint size, const T* indata)
  {
    const uint num_entities = _mesh->num_entities(_entity_dim);
    dolfin_assert(num_entities == size);

    vtkSmartPointer<vtkFloatArray> values =
      vtkSmartPointer<vtkFloatArray>::New();
    values->SetNumberOfValues(num_entities);

    for (uint i = 0; i < num_entities; ++i)
    {
      values->SetValue(i, (float)indata[i]);
    }

    _grid->GetCellData()->SetScalars(values);
  }
  //----------------------------------------------------------------------------
}

#endif

#endif


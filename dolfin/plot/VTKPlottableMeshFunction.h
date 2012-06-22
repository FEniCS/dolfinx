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
// First added:  2012-06-21
// Last changed: 2012-06-21

#ifndef __VTK_PLOTTABLE_MESH_FUNCTION_H
#define __VTK_PLOTTABLE_MESH_FUNCTION_H

#ifdef HAS_VTK

#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkCellData.h>

#include <dolfin/mesh/MeshFunction.h>

namespace dolfin
{

  // Forward declarations
  class VTKPlottableMesh;

  template <typename T> class VTKPlottableMeshFunction : public VTKPlottableMesh
  {
  public:

    explicit VTKPlottableMeshFunction(
        boost::shared_ptr<const MeshFunction<T> > mesh_function);
    
    //--- Implementation of the GenericVTKPlottable interface ---
    
    /// Initialize the parts of the pipeline that this class controls
    void init_pipeline();

    /// Update the plottable data
    void update(const Parameters& parameters);

    /// Update the scalar range of the plottable data
    void update_range(double range[2]);

    /// Return data to visualize
    vtkSmartPointer<vtkAlgorithmOutput> get_output() const;

  private: 

    // Update vertex values
    void update_vertices();

    // Update cell values
    void update_cells();

    // The mesh function
    boost::shared_ptr<const MeshFunction<T> > _mesh_function;

  };

  //---------------------------------------------------------------------------
  // Implementation of VTKPlottableMeshFunction
  //---------------------------------------------------------------------------
  template <typename T>
  VTKPlottableMeshFunction<T>::VTKPlottableMeshFunction(
      boost::shared_ptr<const MeshFunction<T> > mesh_function) :
    VTKPlottableMesh(reference_to_no_delete_pointer(mesh_function->mesh())),
    _mesh_function(mesh_function)
  {
    // Do nothing
  }
  //----------------------------------------------------------------------------
  template <typename T>
  void VTKPlottableMeshFunction<T>::init_pipeline()
  {
    _geometryFilter->SetInput(_grid);
    _geometryFilter->Update();
  }
  //----------------------------------------------------------------------------
  template <typename T>
  void VTKPlottableMeshFunction<T>::update(const Parameters& parameters)
  {
    VTKPlottableMesh::update(parameters);

    if (_mesh_function->dim() == 0) {
      // Vertex valued mesh function 
      update_vertices();
    } else if (_mesh_function->dim() == _mesh->topology().dim()) {
      // Cell valued
      update_cells();
    } else {
      dolfin_error("VTKPlottableMeshFunction.h",
                   "plot mesh function",
                   "Only able to plot vertex and cell valued mesh functions.");
    }
  }
  //----------------------------------------------------------------------------
  template <typename T>
  void VTKPlottableMeshFunction<T>::update_range(double range[2])
  {
    // FIXME: Do we need to define this since it just calls the superclass?
    VTKPlottableMesh::update_range(range);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  vtkSmartPointer<vtkAlgorithmOutput> VTKPlottableMeshFunction<T>::get_output() const
  {
    // FIXME: Do we need to define this since it just calls the superclass?
    return _geometryFilter->GetOutputPort();
  }
  //----------------------------------------------------------------------------
  template <typename T>
  void VTKPlottableMeshFunction<T>::update_vertices()
  {
    dolfin_assert(_mesh_function->dim() == 0);

    // Update vertex/point data

    // Make VTK float array and allocate storage for mesh function values
    uint num_vertices = _mesh->num_vertices();
    vtkSmartPointer<vtkFloatArray> values =
      vtkSmartPointer<vtkFloatArray>::New();
    values->SetNumberOfValues(num_vertices);

    // Iterate the mesh function and convert the value at each vertex to double
    T value;
    for(uint i = 0; i < num_vertices; ++i) {
      value = (*_mesh_function)[i];
      values->SetValue(i, (double) value);
    }

    // Attach scalar values as point data in the VTK grid
    _grid->GetPointData()->SetScalars(values);
  }
  //----------------------------------------------------------------------------
  template <typename T>
  void VTKPlottableMeshFunction<T>::update_cells()
  {
    dolfin_assert(_mesh_function->dim() == _mesh->topology().dim());

    // Update cell data

    // Make VTK float array and allocate storage for mesh function values
    uint num_cells = _mesh->num_cells();
    vtkSmartPointer<vtkFloatArray> values =
      vtkSmartPointer<vtkFloatArray>::New();
    values->SetNumberOfValues(num_cells);

    // Iterate the mesh function and convert the value at each vertex to double
    T value;
    for(uint i = 0; i < num_cells; ++i) {
      value = (*_mesh_function)[i];
      values->SetValue(i, (double) value);
    }

    // Attach scalar values as point data in the VTK grid
    _grid->GetCellData()->SetScalars(values);
  }
  //----------------------------------------------------------------------------

}

#endif

#endif


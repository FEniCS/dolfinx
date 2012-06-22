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

#ifdef HAS_VTK

#include <vtkCellArray.h>
#include <vtkArrowSource.h>
#include <vtkFloatArray.h>
#include <vtkPointData.h>
#include <vtkVectorNorm.h>

#include <dolfin/common/Timer.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/function/FunctionSpace.h>

#include "VTKPlottableMesh.h"
#include "VTKPlottableGenericFunction.h"

using namespace dolfin;

//----------------------------------------------------------------------------
VTKPlottableGenericFunction::VTKPlottableGenericFunction(
    boost::shared_ptr<const Function> function) :
  VTKPlottableMesh(function->function_space()->mesh()),
  _function(function),
  warp_vector_mode(false)
{
  // Do nothing
}
VTKPlottableGenericFunction::VTKPlottableGenericFunction(
    boost::shared_ptr<const Expression> expression,
    boost::shared_ptr<const Mesh> mesh) :
  VTKPlottableMesh(mesh),
  _function(expression),
  warp_vector_mode(false)
{
  // Do nothing
}
//----------------------------------------------------------------------------
void VTKPlottableGenericFunction::init_pipeline()
{
  _warpscalar = vtkSmartPointer<vtkWarpScalar>::New();
  _warpvector = vtkSmartPointer<vtkWarpVector>::New();
  _glyphs = vtkSmartPointer<vtkGlyph3D>::New();

  switch (_function->value_rank()) {
    
    // Setup pipeline for scalar functions
    case 0:
      if (_mesh->topology().dim() < 3) {
        // In 1D and 2D, we warp the mesh according to the scalar values
        _warpscalar->SetInput(_grid);

        _geometryFilter->SetInput(_warpscalar->GetOutput());
      }
      else {
        // In 3D, we just show the scalar values as colors on the mesh
        _geometryFilter->SetInput(_grid);

      }
      _geometryFilter->Update();
      break;
    
    // Setup pipeline for vector functions. Everything is set up except the
    // mapper
    case 1: {
      // Setup pipeline for warp visualization
      _warpvector->SetInput(_grid);
      _geometryFilter->SetInput(_warpvector->GetOutput());
      _geometryFilter->Update();

      // Setup pipeline for glyph visualization
      vtkSmartPointer<vtkArrowSource> arrow =
        vtkSmartPointer<vtkArrowSource>::New();
      arrow->SetTipRadius(0.08);
      arrow->SetTipResolution(16);
      arrow->SetTipLength(0.25);
      arrow->SetShaftRadius(0.05);
      arrow->SetShaftResolution(16);

      // Create the glyph object, set source (the arrow) and input (the grid) and
      // adjust various parameters
      _glyphs->SetSourceConnection(arrow->GetOutputPort());
      _glyphs->SetInput(_grid);
      _glyphs->SetVectorModeToUseVector();
      _glyphs->SetScaleModeToScaleByVector();
      _glyphs->SetColorModeToColorByVector();
      break;
    }
    default:
      dolfin_error("VTKPlotter.cpp",
          "plot function of rank > 2.",
          "Plotting of higher order functions is not supported.");
  }
}
//----------------------------------------------------------------------------
void VTKPlottableGenericFunction::update(const Parameters& parameters)
{
  VTKPlottableMesh::update(parameters);

  const double scale = parameters["scale"];
  _warpscalar->SetScaleFactor(scale);
  _warpvector->SetScaleFactor(scale);
  _glyphs->SetScaleFactor(scale);

  const std::string mode = parameters["mode"];
  if (mode == "warp") {
    warp_vector_mode = true;
  } else {
    if (mode != "auto") {
      warning("Unrecognized mode \"" + mode + "\", using default (glyphs).");
    }
    warp_vector_mode = false;
  }
  
  switch(_function->value_rank()) {
    case 0:
      update_scalar();
      break;
    case 1:
      update_vector();
      break;
    default:
      break;
  }
}
//----------------------------------------------------------------------------
void VTKPlottableGenericFunction::update_range(double range[2])
{
  // Superclass gets the range from the grid
  VTKPlottableMesh::update_range(range);

  // We also multiply the warpscalar's scale factor by the width of the scalar
  // range
  double scale = _warpscalar->GetScaleFactor();
  _warpscalar->SetScaleFactor(scale/(range[1]-range[0]));
}
//----------------------------------------------------------------------------
vtkSmartPointer<vtkAlgorithmOutput> VTKPlottableGenericFunction::get_output() const
{
  // In the 3D glyph case, return the glyphs' output
  if (_function->value_rank() == 1 && !warp_vector_mode) {
    return _glyphs->GetOutputPort();
  // Otherwise return the geometryfilter's output
  } else {
    return _geometryFilter->GetOutputPort();
  }
}
//----------------------------------------------------------------------------
void VTKPlottableGenericFunction::update_scalar()
{
  dolfin_assert(_function->value_rank() == 0);

  // Update scalar point data

  // Make VTK float array and allocate storage for function values
  uint num_vertices = _mesh->num_vertices();
  vtkSmartPointer<vtkFloatArray> scalars =
    vtkSmartPointer<vtkFloatArray>::New();
  scalars->SetNumberOfValues(num_vertices);

  // Evaluate DOLFIN function and copy values to the VTK array
  std::vector<double> vertex_values(num_vertices);
  _function->compute_vertex_values(vertex_values, *_mesh);

  for(uint i = 0; i < num_vertices; ++i) {
    scalars->SetValue(i, vertex_values[i]);
  }

  // Attach scalar values as point data in the VTK grid
  _grid->GetPointData()->SetScalars(scalars);
}
//----------------------------------------------------------------------------
void VTKPlottableGenericFunction::update_vector()
{
  dolfin_assert(_function->value_rank() == 1);

  // Update vector and scalar point data

  // Make VTK float array and allocate storage for function vector values
  uint num_vertices = _mesh->num_vertices();
  uint num_components = _function->value_dimension(0);
  vtkSmartPointer<vtkFloatArray> vectors =
    vtkSmartPointer<vtkFloatArray>::New();

  // NOTE: Allocation must be done in this order!
  // Note also that the number of VTK vector components must always be 3
  // regardless of the function vector value dimension
  vectors->SetNumberOfComponents(3);
  vectors->SetNumberOfTuples(num_vertices);

  // Evaluate DOLFIN function and copy values to the VTK array
  // The entries in "vertex_values" must be copied to "vectors". Viewing
  // these arrays as matrices, the transpose of vertex values should be copied,
  // since DOLFIN and VTK store vector function values differently
  std::vector<double> vertex_values(num_vertices*num_components);
  _function->compute_vertex_values(vertex_values, *_mesh);

  for(uint i = 0; i < num_vertices; ++i) {
    vectors->SetValue(3*i,     vertex_values[i]);
    vectors->SetValue(3*i + 1, vertex_values[i + num_vertices]);

    // If the DOLFIN function vector value dimension is 2, pad with a 0
    if(num_components == 2) {
      vectors->SetValue(3*i + 2, 0.0);
      // else, add the last entry in the value vector
    } else {
      vectors->SetValue(3*i + 2, vertex_values[i + num_vertices*2]);
    }
  }
  // Attach vectors as vector point data in the VTK grid
  _grid->GetPointData()->SetVectors(vectors);

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
//----------------------------------------------------------------------------
#endif

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
// Last changed: 2012-09-19

#ifdef HAS_VTK

#include <vtkArrowSource.h>
#include <vtkUnstructuredGrid.h>

#include <dolfin/common/Timer.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Vertex.h>

#include "VTKPlottableMesh.h"
#include "VTKPlottableGenericFunction.h"
#include "VTKPlottableGenericFunction1D.h"
#include "ExpressionWrapper.h"

using namespace dolfin;

//----------------------------------------------------------------------------
VTKPlottableGenericFunction::VTKPlottableGenericFunction(boost::shared_ptr<const Function> function)
  : VTKPlottableMesh(function->function_space()->mesh()),
    _function(function)
{
  // Do nothing
}
//----------------------------------------------------------------------------
VTKPlottableGenericFunction::VTKPlottableGenericFunction(boost::shared_ptr<const Expression> expression,
                                                         boost::shared_ptr<const Mesh> mesh)
  : VTKPlottableMesh(mesh),
    _function(expression)
{
  // Do nothing
}
//----------------------------------------------------------------------------
uint VTKPlottableGenericFunction::value_rank() const
{
  return _function->value_rank();
}
//----------------------------------------------------------------------------
void VTKPlottableGenericFunction::init_pipeline(const Parameters &parameters)
{
  _warpscalar = vtkSmartPointer<vtkWarpScalar>::New();
  _warpvector = vtkSmartPointer<vtkWarpVector>::New();
  _glyphs = vtkSmartPointer<vtkGlyph3D>::New();

  VTKPlottableMesh::init_pipeline(parameters);

  _mode = (std::string)parameters["mode"];

  switch (_function->value_rank())
  {
    // Setup pipeline for scalar functions
  case 0:
    {
      if (mesh()->topology().dim() < 3)
      {
        // In 1D and 2D, we warp the mesh according to the scalar values
        if (_mode != "off")
        {
          insert_filter(_warpscalar);
        }
      }
      else
      {
        // In 3D, we just show the scalar values as colors on the mesh
      }
    }
    break;

    // Setup pipeline for vector functions. Everything is set up except the
    // mapper
  case 1:
    {
      // Setup pipeline for warp visualization
      if (_mode != "off")
      {
        insert_filter(_warpvector);
      }

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
      _glyphs->SetInput(grid());
      _glyphs->SetVectorModeToUseVector();
      _glyphs->SetScaleModeToScaleByVector();
      _glyphs->SetColorModeToColorByVector();
    }
    break;
  default:
    {
    dolfin_error("VTKPlotter.cpp",
                 "plot function of rank > 2.",
                 "Plotting of higher order functions is not supported.");
    }
  }
}
//----------------------------------------------------------------------------
bool VTKPlottableGenericFunction::is_compatible(const Variable &var) const
{
  const GenericFunction *function(dynamic_cast<const Function*>(&var));
  const ExpressionWrapper *wrapper(dynamic_cast<const ExpressionWrapper*>(&var));
  const Mesh *mesh(NULL);

  if (function)
  {
    mesh = static_cast<const Function*>(function)->function_space()->mesh().get();
  }
  else if (wrapper)
  {
    function = wrapper->expression().get();
    mesh = wrapper->mesh().get();
  }
  else
  {
    return false;
  }
  if (function->value_rank() > 1 || (function->value_rank() == 0) != !_glyphs->GetInput())
  {
    return false;
  }
  return VTKPlottableMesh::is_compatible(*mesh);
}
//----------------------------------------------------------------------------
void VTKPlottableGenericFunction::update(boost::shared_ptr<const Variable> var, const Parameters& parameters, int frame_counter)
{
  boost::shared_ptr<const Mesh> mesh = VTKPlottableMesh::mesh();
  if (var)
  {
    boost::shared_ptr<const Function> function(boost::dynamic_pointer_cast<const Function>(var));
    boost::shared_ptr<const ExpressionWrapper> wrapper(boost::dynamic_pointer_cast<const ExpressionWrapper>(var));
    dolfin_assert(function || wrapper);
    if (function)
    {
      mesh = function->function_space()->mesh();
      _function = function;
    }
    else
    {
      mesh = wrapper->mesh();
      _function = wrapper->expression();
    }
  }

  // Update the mesh
  VTKPlottableMesh::update(mesh, parameters, frame_counter);

  // Update the values on the mesh
  const Function *func = dynamic_cast<const Function *>(_function.get());
  if (func && func->vector()->local_size() == (uint)grid()->GetNumberOfCells())
  {
    // Hack to display DG0 functions. Should really be implemented using
    // duplicate points (one point per vertex per cell), so that warping
    // and DG1 work as expected.
    // Also: How do we find out if a FunctionSpace is discontinuous?
    insert_filter(NULL); // expel the warpscalar filter
    std::vector<double> cell_values;
    func->vector()->get_local(cell_values);
    setCellValues(cell_values.size(), &cell_values[0], parameters);
  }
  else
  {
    std::vector<double> vertex_values;
    _function->compute_vertex_values(vertex_values, *mesh);
    setPointValues(vertex_values.size(), &vertex_values[0], parameters);
  }
}
//----------------------------------------------------------------------------
void VTKPlottableGenericFunction::rescale(double range[2], const Parameters &parameters)
{
  const double scale = parameters["scale"];
  _warpvector->SetScaleFactor(scale);
  _glyphs->SetScaleFactor(scale);

  // Compute the scale factor for scalar warping
  double* bounds = grid()->GetBounds();
  double grid_h = std::max(bounds[1]-bounds[0], bounds[3]-bounds[2]);

  // Set the default warp such that the max warp is one fourth of the longest
  // axis of the mesh.
  double scale_factor = grid_h / 4.0;
  if (range[1] > range[0])
  {
    scale_factor /= (range[1] - range[0]);
  }
  _warpscalar->SetScaleFactor(scale * scale_factor);
}
//----------------------------------------------------------------------------
void VTKPlottableGenericFunction::update_range(double range[2])
{
  // Superclass gets the range from the grid
  VTKPlottableMesh::update_range(range);
}
//----------------------------------------------------------------------------
vtkSmartPointer<vtkAlgorithmOutput> VTKPlottableGenericFunction::get_output() const
{
  // In the 3D glyph case, return the glyphs' output
  if (_function->value_rank() == 1 && _mode != "warp")
  {
    return _glyphs->GetOutputPort();
  }
  else
  {
    return VTKPlottableMesh::get_output();
  }
}
//----------------------------------------------------------------------------
VTKPlottableGenericFunction *dolfin::CreateVTKPlottable(boost::shared_ptr<const Function> function)
{
  if (function->function_space()->mesh()->topology().dim() == 1)
  {
    return new VTKPlottableGenericFunction1D(function);
  }
  return new VTKPlottableGenericFunction(function);
}
//----------------------------------------------------------------------------
VTKPlottableGenericFunction *dolfin::CreateVTKPlottable(boost::shared_ptr<const ExpressionWrapper> wrapper)
{
  return CreateVTKPlottable(wrapper->expression(), wrapper->mesh());
}
//----------------------------------------------------------------------------
VTKPlottableGenericFunction *dolfin::CreateVTKPlottable(boost::shared_ptr<const Expression> expr,
                                                        boost::shared_ptr<const Mesh> mesh)
{
  if (mesh->topology().dim() == 1)
  {
    return new VTKPlottableGenericFunction1D(expr, mesh);
  }
  return new VTKPlottableGenericFunction(expr, mesh);
}

#endif

// Copyright (C) 2012 Joachim B Haga.
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
// First added:  2012-09-11
// Last changed: 2012-09-13

#ifndef __VTK_PLOTTABLE_GENERICFUNCTION_1D_H
#define __VTK_PLOTTABLE_GENERICFUNCTION_1D_H

#ifdef HAS_VTK

#include <vtkSmartPointer.h>

#include <dolfin/mesh/Mesh.h>
#include "VTKPlottableGenericFunction.h"

class vtkXYPlotActor;

namespace dolfin
{

  class Mesh;
  class VTKWindowOutputStage;

  ///

  class VTKPlottableGenericFunction1D : public VTKPlottableGenericFunction
  {
  public:

    explicit
    VTKPlottableGenericFunction1D(boost::shared_ptr<const Function> function);

    explicit
    VTKPlottableGenericFunction1D(boost::shared_ptr<const Expression> expression,
                                  boost::shared_ptr<const Mesh> mesh);

    //--- Implementation of the GenericVTKPlottable interface ---

    /// Additional parameters for VTKPlottableGenericFunction1D
    virtual void modify_default_parameters(Parameters &parameters)
    {
      parameters["scalarbar"] = false;
    }

    /// Initialize the parts of the pipeline that this class controls
    virtual void init_pipeline(const Parameters &parameters);

    /// Connect or reconnect to the output stage.
    virtual void connect_to_output(VTKWindowOutputStage& output);

    /// Update the plottable data
    virtual void update(boost::shared_ptr<const Variable> var, const Parameters& parameters, int frame_counter);

    /// Inform the plottable about the range.
    virtual void rescale(double range[2], const Parameters& parameters);

    /// Return whether this plottable is compatible with the variable
    virtual bool is_compatible(const Variable &var) const;

    /// Get an actor for showing vertex labels
    virtual vtkSmartPointer<vtkActor2D> get_vertex_label_actor(vtkSmartPointer<vtkRenderer>);

    /// Get an actor for showing cell labels
    virtual vtkSmartPointer<vtkActor2D> get_cell_label_actor(vtkSmartPointer<vtkRenderer>);

  private:

    vtkSmartPointer<vtkXYPlotActor> _actor;

  };

}

#endif

#endif

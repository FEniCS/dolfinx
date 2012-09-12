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
// First added:  2012-06-21
// Last changed: 2012-09-12

#ifndef __VTK_PLOTTABLE_MESH_FUNCTION_H
#define __VTK_PLOTTABLE_MESH_FUNCTION_H

#ifdef HAS_VTK

#include "VTKPlottableMesh.h"

namespace dolfin
{

  // Forward declarations
  template<typename T> class MeshFunction;

  template <typename T> class VTKPlottableMeshFunction : public VTKPlottableMesh
  {
  public:

    explicit
    VTKPlottableMeshFunction(boost::shared_ptr<const MeshFunction<T> > mesh_function);

    //--- Implementation of the GenericVTKPlottable interface ---

    /// Additional parameters for VTKPlottableMeshFunction
    virtual Parameters default_parameters()
    {
      return Parameters();
    }

    /// Update the plottable data
    void update(boost::shared_ptr<const Variable> var, const Parameters& parameters, int frame_counter);

    bool is_compatible(const Variable &var) const { return dynamic_cast<const MeshFunction<T>*>(&var); }

  private:

    // The mesh function
    boost::shared_ptr<const MeshFunction<T> > _mesh_function;

  };

  //----------------------------------------------------------------------------

  template <typename T>
  VTKPlottableMeshFunction<T> *CreateVTKPlottable(boost::shared_ptr<const MeshFunction<T> > meshfunc)
  {
    return new VTKPlottableMeshFunction<T>(meshfunc);
  }

  template <typename T>
  VTKPlottableMeshFunction<T> *CreateVTKPlottable(boost::shared_ptr<MeshFunction<T> > meshfunc)
  {
    return new VTKPlottableMeshFunction<T>(meshfunc);
  }

}

#endif

#endif

// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-03-31
// Last changed: 2005

#include <dolfin/constants.h>
#include <dolfin/Matrix.h>

namespace dolfin
{

  /// The standard stiffness matrix for homogeneous Neumann
  /// boundary conditions on a given mesh.

  class StiffnessMatrix : public Matrix
  {
  public:
  
    /// Construct a stiffness matrix for a given mesh
    StiffnessMatrix(Mesh& mesh, real epsilon = 1.0);

  };

}

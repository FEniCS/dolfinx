// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

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

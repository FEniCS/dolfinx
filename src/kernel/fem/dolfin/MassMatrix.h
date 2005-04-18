// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Matrix.h>

namespace dolfin
{

  /// The standard mass matrix on a given mesh.

  class MassMatrix : public Matrix
  {
  public:
  
    /// Construct a mass matrix for a given mesh
    MassMatrix(Mesh& mesh);

  };

}

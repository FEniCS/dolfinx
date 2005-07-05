// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-03-31
// Last changed: 2005

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

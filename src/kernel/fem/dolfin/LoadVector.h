// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-03-31
// Last changed: 2005

#include <dolfin/Vector.h>

namespace dolfin
{

  /// The standard load vector on a given mesh.

  class LoadVector : public Vector
  {
  public:
  
    /// Construct a load vector for a given mesh
    LoadVector(Mesh& mesh);

  };

}

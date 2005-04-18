// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

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

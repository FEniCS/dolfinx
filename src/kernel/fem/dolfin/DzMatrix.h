// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/constants.h>
#include <dolfin/Matrix.h>
#include <dolfin/PDE.h>

namespace dolfin
{

  /// A linear mapping for computing an averaged value of the derivative
  /// in the z-direction of a piecewise linear function represented in
  /// terms of its nodal values on a given mesh.
  ///
  /// Since the derivative is not defined at the nodes, the derivative is
  /// computed as the L2 projection onto the test/trial space (with a lumped
  /// mass matrix).
  ///
  /// Simple usage:
  ///
  /// Vector x;
  /// Function u(mesh, x);
  /// DzMatrix Dz(mesh);
  /// real dudz = Dz.mult(x, i);

  class DzMatrix : public Matrix
  {
  public:
  
    /// Construct linear mapping for z-derivative on a given mesh
    DzMatrix(Mesh& mesh);
    
  };
  
}

// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/constants.h>
#include <dolfin/Matrix.h>
#include <dolfin/PDE.h>

namespace dolfin
{

  /// A linear mapping for computing an averaged value of the derivative
  /// in the x-direction of a piecewise linear function represented in
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
  /// DxMatrix Dx(mesh);
  /// real dudx = Dx.mult(x, i);

  class DxMatrix : public Matrix
  {
  public:
  
    /// Construct linear mapping for x-derivative on a given mesh
    DxMatrix(Mesh& mesh);
    
  };
  
}

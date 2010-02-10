// Copyright (C) 2006-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristian B. Oelgaard, 2007.
// Modified by Martin Sandve Alnes, 2008.
// Modified by Garth N. Wells, 2008.
//
// First added:  2006-02-09
// Last changed: 2009-10-05

#ifndef __SPECIAL_FUNCTIONS_H
#define __SPECIAL_FUNCTIONS_H

#include "dolfin/common/Array.h"
#include "Expression.h"

namespace dolfin
{

  class Mesh;

  /// This Function represents the mesh coordinates on a given mesh.
  class MeshCoordinates : public Expression
  {
  public:

    /// Constructor
    MeshCoordinates(const Mesh& mesh);

    /// Evaluate function
    void eval(Array<double>& values, const Data& data) const;

  private:

    // The mesh
    const Mesh& mesh;

  };

  /// This Function represents the local cell size on a given mesh.
  class CellSize : public Expression
  {
  public:

    /// Constructor
    CellSize(const Mesh& mesh);

    /// Evaluate function
    void eval(Array<double>& values, const Data& data) const;

  private:

    // The mesh
    const Mesh& mesh;

  };

  /// This function represents the area/length of a cell facet on a given mesh.
  class FacetArea : public Expression
  {
  public:

    /// Constructor
    FacetArea(const Mesh& mesh);

    /// Evaluate function
    void eval(Array<double>& values, const Data& data) const;

  private:

    // The mesh
    const Mesh& mesh;

  };

}

#endif

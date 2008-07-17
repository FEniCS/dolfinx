// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristian B. Oelgaard, 2007.
// Modified by Martin Sandve Alnes, 2008.
// Modified by Garth N. Wells, 2008.
//
// First added:  2006-02-09
// Last changed: 2008-07-17

#ifndef __SPECIAL_FUNCTIONS_H
#define __SPECIAL_FUNCTIONS_H

#include "Function.h"

namespace dolfin
{

  class Form;
  class UFC;


  /// This Function represents the local mesh size on a given mesh.
  class MeshSize : public Function
  {
  public:

    /// Constructor
    MeshSize(Mesh& mesh);

    /// Return cell size
    real eval(const real* x) const;
    
    /// Compute minimal cell diameter
    real min() const;

    /// Compute maximal cell diameter
    real max() const;    
  };

  /// This Function represents the inverse of the local cell size on a given 
  /// mesh.
  class InvMeshSize : public Function
  {
  public:

    /// Constructor
    InvMeshSize(Mesh& mesh);

    /// Return inverse of cell size
    real eval(const real* x) const;
  };

  /// This Function represents the average of the local cell size (average of 
  /// cell sharing a facet) on a given mesh.
  class AvgMeshSize : public Function
  {
  public:

    /// Constructor
    AvgMeshSize(Mesh& mesh);

    /// Return average cell size
    real eval(const real* x) const;
  };

  /// This Function represents the outward unit normal on cell facets.
  /// Note that it is only nonzero on cell facets (not on cells).
  class FacetNormal : public Function
  {
  public:

    FacetNormal(Mesh& mesh);

    void eval(real* values, const real* x) const;

    uint rank() const;
    
    uint dim(uint i) const;
  };

  /// This function represents the area/length of a cell facet.
  class FacetArea : public Function
  {
  public:

    FacetArea(Mesh& mesh);

    void eval(real* values, const real* x) const;
  };

  /// This function represents the inverse area/length of a cell facet.
  class InvFacetArea : public Function
  {
  public:

    InvFacetArea(Mesh& mesh);

    void eval(real* values, const real* x) const;
  };

  /// This function determines if the current facet is an outflow facet with
  /// respect to the current cell. It accepts as argument the mesh and a form
  /// M = dot(n, v)*ds, a functional, defined on the normal vector to the
  /// facet and velocity vector integrated over the exterior of the cell.
  /// The function returns 1.0 if the dot product > 0, 0.0 otherwise.
  class OutflowFacet : public Function
  {
  public:

    // Constructor
    OutflowFacet(Mesh& mesh, Form& form);

    ~OutflowFacet();

    real eval(const real* x) const;

  private:

    Form& form;
    UFC* ufc;
  };

}

#endif

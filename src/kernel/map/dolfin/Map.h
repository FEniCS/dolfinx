// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __MAP_H
#define __MAP_H

#include <dolfin/Point.h>
#include <dolfin/ShapeFunction.h>
#include <dolfin/Product.h>
#include <dolfin/ElementFunction.h>

namespace dolfin
{
  
  class Cell;
  class Edge;
  class Face;

  /// Map from reference cell to actual cell, including the
  /// derivative of the map and the inverse of the derivative.
  ///
  /// It is assumed that the map is affine (constant determinant)
  /// and that the reference cells are given by
  ///
  ///   (0) - (1)                             in 1D
  ///
  ///   (0,0) - (1,0)                         in 2D
  ///
  ///   (0,0,0) - (1,0,0) - (0,1,0) - (0,0,1) in 3D
  
  class Map
  {
  public:
    
    // Constructor
    Map();

    // Destructor
    virtual ~Map();
    
    /// Evaluate map from reference cell
    virtual Point operator() (const Point& p) const = 0;
    
    /// Evaluate map from reference boundary of reference cell to
    /// the given boundary on the current cell
    virtual Point operator() (const Point& p, unsigned int boundary) const = 0;

    /// Return determinant of derivative of map to interior of cell
    real det() const;

    /// Return determinant of derivative of map to boundary of cell
    real bdet() const;

    /// Return current boundary (triangle or tetrahedron) of cell
    int boundary() const;

    /// Return current cell
    const Cell& cell() const;

    /// Update map to given cell
    virtual void update(const Cell& cell) = 0;

    /// Update map to cell of given edge
    virtual void update(const Edge& edge);

    /// Update map to cell of given face
    virtual void update(const Face& face);
    
    /// Return derivative of constant
    real ddx(real a) const;
    real ddy(real a) const;
    real ddz(real a) const;
    real ddt(real a) const;
    
    /// Return derivative of shape function
    virtual const FunctionSpace::ElementFunction ddx(const FunctionSpace::ShapeFunction& v) const = 0;
    virtual const FunctionSpace::ElementFunction ddy(const FunctionSpace::ShapeFunction& v) const = 0;
    virtual const FunctionSpace::ElementFunction ddz(const FunctionSpace::ShapeFunction& v) const = 0;
    virtual const FunctionSpace::ElementFunction ddt(const FunctionSpace::ShapeFunction& v) const = 0;
    
  protected:
    
    void reset();
    
    int dim; // Dimension
    
    // Jacobian of map from reference cell
    real f11, f12, f13;
    real f21, f22, f23;
    real f31, f32, f33;
    
    // Inverse of F
    real g11, g12, g13;
    real g21, g22, g23;
    real g31, g32, g33;
    
    // Determinant of derivative of map to cell
    real d;

    // Determinant of derivative of map to boundary of cell
    real bd;

    // Current boundary (triangle or tetrahedron) of cell
    int _boundary;

    // Current cell
    const Cell* _cell;
  };
  
}

#endif

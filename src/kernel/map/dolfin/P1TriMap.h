// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Map from the reference cell defined by the nodes
// (0,0) - (1,0) - (0,1) to a given triangle in 2D.

#ifndef __P1_TRI_MAP_H
#define __P1_TRI_MAP_H

#include <dolfin/Map.h>

namespace dolfin {
  
  class P1TriMap : public Map
  {
  public:
    
    P1TriMap();

    Point operator() (const Point& p) const;
    Point operator() (const Point& p, unsigned int boundary) const;
    
    void update(const Cell& cell);
    void update(const Edge& edge);

    const FunctionSpace::ElementFunction ddx(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction ddy(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction ddz(const FunctionSpace::ShapeFunction &v) const;
    const FunctionSpace::ElementFunction ddt(const FunctionSpace::ShapeFunction &v) const;

  private:

    // FIXME: Temporary hack until we sort things out with the
    // FIXME: local numbering of nodes, cells, edges, and faces.
    unsigned int edgeNumber(const Edge& edge, const Cell& cell) const;
  
  };

}

#endif

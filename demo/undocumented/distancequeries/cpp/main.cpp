// =====================================================================================
//
// Copyright (C) 2010-03-03  André Massing
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by André Massing, 2010
//
// First added:  2010-03-03
// Last changed: 2010-03-04
// 
//Author:  André Massing (am), massing@simula.no
//Company:  Simula Research Laboratory, Fornebu, Norway
//
//Description:
//This demo program demonstrates how to do point distance queries, i.e. 
//how to obtain the point in the mesh or the cell in the mesh or both which
//are nearest to a given point.
// =====================================================================================

#include <dolfin.h>
#include <vector>
#include <utility>

using namespace dolfin;

typedef std::vector<Point> PointList;
typedef PointList::const_iterator PointListIter;


#ifdef HAS_CGAL

int main()
{

  //Rectangle mesh stretching from (-1,-1,0) to (1,1,0) with 20 sample points each axis.
  Rectangle mesh(-1,-1,1,1,20,20);

  PointList point_list;

  //First row along  y = -1.5
  point_list.push_back(Point(-1.5,-1.5,0.0));
  point_list.push_back(Point(0.0,-1.5,0.0));
  point_list.push_back(Point(1.5,-1.5,0.0));

  //Second row along  y = 0;
  point_list.push_back(Point(-1.5,0.0,0.0));
  point_list.push_back(Point(0.0,0.0,0.0));
  point_list.push_back(Point(1.5,0.0,0.0));

  //Third row along  y = 1.5;
  point_list.push_back(Point(-1.5,1.5,0.0));
  point_list.push_back(Point(0.0,1.5,0.0));
  point_list.push_back(Point(1.5,1.5,0.0));

  //Queries for nearest point and cell.
  for (PointListIter i = point_list.begin(); i != point_list.end(); ++i)
  {
    //Obtain nearest point
    cout <<"Nearest point for Point "
    << *i << " is Point " << mesh.closest_point(*i) << endl;
    cout <<"Nearest cell for Point "
    << *i << " is cell " << mesh.closest_cell(*i) << endl;

    //Repetion with closest_point_and_index
    cout <<"Repetition obtaining both nearest point and cell at once:" <<endl; 
    std::pair<Point, dolfin::uint> pc(mesh.closest_point_and_cell(*i));
    cout <<"Nearest point for Point "
    << *i << " is Point " << pc.first << endl;
    cout <<"Nearest cell for Point "
    << *i << " is cell " << pc.second << endl <<endl;
  }

return 0;

}

#else

int main()
{
  info("DOLFIN must be compiled with CGAL to run this demo.");
  return 0;
}

#endif

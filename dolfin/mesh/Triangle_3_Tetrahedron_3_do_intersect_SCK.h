// Copyright (C) 2009 Andre Massing 
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-09-11
// Last changed: 2009-11-29

#ifndef CGAL_TRIANGLE_3_TETRAHEDRON_3_DO_INTERSECT_SCK_H
#define CGAL_TRIANGLE_3_TETRAHEDRON_3_DO_INTERSECT_SCK_H

#include <CGAL/Triangle_3_Triangle_3_do_intersect.h>
#include <CGAL/enum.h>

CGAL_BEGIN_NAMESPACE

template <>
inline bool do_intersect<CGAL::Simple_cartesian<double> > (const Tetrahedron_3<CGAL::Simple_cartesian<double> > &tet,
			 const Triangle_3<CGAL::Simple_cartesian<double> > &tr)
{
  typedef  CGAL::Simple_cartesian<double>::Triangle_3 Triangle;

  if (!tet.has_on_unbounded_side(tr[0])) return true;
  if (!tet.has_on_unbounded_side(tr[1])) return true;
  if (!tet.has_on_unbounded_side(tr[2])) return true;

  if (do_intersect(tr, Triangle(tet[0], tet[1], tet[2]))) return true;
  if (do_intersect(tr, Triangle(tet[0], tet[1], tet[3]))) return true;
  if (do_intersect(tr, Triangle(tet[0], tet[2], tet[3]))) return true;
  if (do_intersect(tr, Triangle(tet[1], tet[2], tet[3]))) return true;
  
  return false;
}

template <>
inline bool do_intersect<CGAL::Simple_cartesian<double> > (const Triangle_3<CGAL::Simple_cartesian<double> > &tr,
			 const Tetrahedron_3<CGAL::Simple_cartesian<double> > &tet)
{
  return do_intersect<CGAL::Simple_cartesian<double> > (tet,tr);
}

CGAL_END_NAMESPACE

#endif // CGAL_TRIANGLE_3_TETRAHEDRON_3_DO_INTERSECT_SCK_H

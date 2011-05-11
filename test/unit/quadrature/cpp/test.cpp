//
// Copyright (C) 2010 Andre Massing
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Andre Massing, 2010
//
// First added:  2010-04-01
// Last changed: 2010-04-09
// 
//Author:  Andre Massing (am), massing@simula.no
//Company:  Simula Research Laboratory, Fornebu, Norway
//
//Description: Unittest for BaryCenterQuadrature. =====================================================================================

#include<vector>
#include <iostream>

#include <dolfin.h>
#include <dolfin/common/unittest.h>

#ifdef HAS_CGAL

#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/Polyhedron_3.h>


using namespace dolfin;

typedef Nef_polyhedron_3::Aff_transformation_3 Aff_transformation_3;
typedef Nef_polyhedron_3::Plane_3 Plane_3;
typedef Nef_polyhedron_3::Vector_3 Vector_3;
typedef Nef_polyhedron_3::Point_3 Point_3;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron_3;

typedef std::vector<Nef_polyhedron_3> PolyhedronList;
typedef PolyhedronList::const_iterator PolyhedronListIterator;

typedef std::vector<int> IntList;
typedef std::vector<int>::const_iterator IntListIterator;
typedef std::vector<double> DoubleList;
typedef std::vector<double>::const_iterator DoubleListIterator;
typedef std::vector<Point> PointList;
typedef std::vector<Point>::const_iterator PointListIterator;

class BaryCenter : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(BaryCenter);
  CPPUNIT_TEST(testSimplePolyhedrons);
  CPPUNIT_TEST(testSimplePolygons);
  CPPUNIT_TEST(testComplexPolyhedrons);
  CPPUNIT_TEST(testComplexPolygons);
  CPPUNIT_TEST_SUITE_END();


  void almost_equal_points(const Point & p1, const Point & p2, double delta)
  {
    CPPUNIT_ASSERT_DOUBLES_EQUAL(p1.x(),p2.x(),delta);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(p1.y(),p2.y(),delta);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(p1.z(),p2.z(),delta);
  }

  //Helper function to create reference polyhedrons.
  void add_test_polyhedron(const Point_3 & p1,
			   const Point_3 & p2, 
			   const Point_3 & p3, 
			   const Point_3 & p4,
			   PolyhedronList & polyhedrons
			  )
  {
    Polyhedron_3 P;
    P.make_tetrahedron(p1,p2,p3,p4);
    Nef_polyhedron_3 N(P);
    polyhedrons.push_back(N);
  }

  void add_test_polyhedron(const Point_3 & p1,
			   const Point_3 & p2, 
			   const Point_3 & p3, 
			   PolyhedronList & polyhedrons
			  )
  {
    Polyhedron_3 P;
    P.make_triangle(p1,p2,p3);
    Nef_polyhedron_3 N(P);
    polyhedrons.push_back(N);
  }

  //Helper function to union disjoint polyhedrons and to compute the volume and barycenter.
  //Indices indicate which polyhedrons should be unioned. No checks at all (index, disjointness etc)
  //Computed polyhedrons, volumes and barycenters will be append to the given list.
  void add_disjoint_polyhedrons(const IntList indices, 
				PolyhedronList & polyhedrons, 
				DoubleList & volumes,
				PointList & points)
  {
    double volume = 0;
    Point point(0,0,0);
    Nef_polyhedron_3 polyhedron;

    for (IntListIterator i = indices.begin(); i != indices.end(); ++i)
    {
      polyhedron += polyhedrons[*i];
      point += volumes[*i] * points[*i]  ;
      volume += volumes[*i];
    }
    point /= volume;

    polyhedrons.push_back(polyhedron);
    volumes.push_back(volume);
    points.push_back(point);
  }


  void testSimplePolyhedrons()
  {
    //Create origin and unit vectors.
    Point_3 e0(0.0, 0.0, 0.0);

    Point_3 e1(1.0, 0.0, 0.0);
    Point_3 e2(0.0, 1.0, 0.0);
    Point_3 e3(0.0, 0.0, 1.0);

    Point_3 _e1(-1.0, 0.0, 0.0);
    Point_3 _e2(0.0, -1.0, 0.0);
    Point_3 _e3(0.0, 0.0, -1.0);

    //Create tetrahedrons with unit vectors and reference results.
    DoubleList reference_volumes;
    PointList reference_bary_centers;
    PolyhedronList reference_polyhedrons;

    add_test_polyhedron(e0,e1,e2,e3,reference_polyhedrons);
    reference_volumes.push_back(1.0/6.0);
    reference_bary_centers.push_back(Point(0.25,0.25,0.25));

    add_test_polyhedron(e0,_e1,e2,e3,reference_polyhedrons);
    reference_volumes.push_back(1.0/6.0);
    reference_bary_centers.push_back(Point(-0.25,0.25,0.25));

    add_test_polyhedron(e0,e1,_e2,e3,reference_polyhedrons);
    reference_volumes.push_back(1.0/6.0);
    reference_bary_centers.push_back(Point(0.25,-0.25,0.25));

    add_test_polyhedron(e0,_e1,_e2,e3,reference_polyhedrons);
    reference_volumes.push_back(1.0/6.0);
    reference_bary_centers.push_back(Point(-0.25,-0.25,0.25));

    add_test_polyhedron(e0,e1,e2,_e3,reference_polyhedrons);
    reference_volumes.push_back(1.0/6.0);
    reference_bary_centers.push_back(Point(0.25,0.25,-0.25));

    add_test_polyhedron(e0,_e1,e2,_e3,reference_polyhedrons);
    reference_volumes.push_back(1.0/6.0);
    reference_bary_centers.push_back(Point(-0.25,0.25,-0.25));

    add_test_polyhedron(e0,e1,_e2,_e3,reference_polyhedrons);
    reference_volumes.push_back(1.0/6.0);
    reference_bary_centers.push_back(Point(0.25,-0.25,-0.25));

    add_test_polyhedron(e0,_e1,_e2,_e3,reference_polyhedrons);
    reference_volumes.push_back(1.0/6.0);
    reference_bary_centers.push_back(Point(-0.25,-0.25,-0.25));

    //Add sum of polyhedrons
    IntList add_indices;
    add_indices.push_back(0);

    add_indices.push_back(1);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(2);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(3);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(4);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(5);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(6);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(7);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    //Add translated version 
    //Upper halfspace
    Nef_polyhedron_3 polyhedron = reference_polyhedrons[0];
    polyhedron.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(1, 1, 1)));
    reference_polyhedrons.push_back(polyhedron);
    reference_volumes.push_back(1.0/6.0);
    reference_bary_centers.push_back(Point(1.25,1.25,1.25));

    polyhedron = reference_polyhedrons[1];
    polyhedron.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(-1, 1, 1)));
    reference_polyhedrons.push_back(polyhedron);
    reference_volumes.push_back(1.0/6.0);
    reference_bary_centers.push_back(Point(-1.25,1.25,1.25));

    polyhedron = reference_polyhedrons[2];
    polyhedron.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(1, -1, 1)));
    reference_polyhedrons.push_back(polyhedron);
    reference_volumes.push_back(1.0/6.0);
    reference_bary_centers.push_back(Point(1.25,-1.25,1.25));

    polyhedron = reference_polyhedrons[3];
    polyhedron.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(-1, -1, 1)));
    reference_polyhedrons.push_back(polyhedron);
    reference_volumes.push_back(1.0/6.0);
    reference_bary_centers.push_back(Point(-1.25,-1.25,1.25));

    polyhedron = reference_polyhedrons[4];
    polyhedron.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(1, 1, -1)));
    reference_polyhedrons.push_back(polyhedron);
    reference_volumes.push_back(1.0/6.0);
    reference_bary_centers.push_back(Point(1.25,1.25,-1.25));

    polyhedron = reference_polyhedrons[5];
    polyhedron.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(-1, 1, -1)));
    reference_polyhedrons.push_back(polyhedron);
    reference_volumes.push_back(1.0/6.0);
    reference_bary_centers.push_back(Point(-1.25,1.25,-1.25));

    polyhedron = reference_polyhedrons[6];
    polyhedron.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(1, -1, -1)));
    reference_polyhedrons.push_back(polyhedron);
    reference_volumes.push_back(1.0/6.0);
    reference_bary_centers.push_back(Point(1.25,-1.25,-1.25));

    polyhedron = reference_polyhedrons[7];
    polyhedron.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(-1, -1, -1)));
    reference_polyhedrons.push_back(polyhedron);
    reference_volumes.push_back(1.0/6.0);
    reference_bary_centers.push_back(Point(-1.25,-1.25,-1.25));

    //Add the disjoint union of the translated polyhedrons.
    add_indices.clear();
    add_indices.push_back(15);

    add_indices.push_back(16);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(17);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(18);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(19);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(20);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(21);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(22);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);


    //Check volume and barycenter for polyhedrons
    for (dolfin::uint i = 0; i < reference_polyhedrons.size(); ++i)
    {
      BarycenterQuadrature quadrature_rule(reference_polyhedrons[i]);
      CPPUNIT_ASSERT_DOUBLES_EQUAL(reference_volumes[i],
				   quadrature_rule.weights()[0], 1.0e-12);
      almost_equal_points(reference_bary_centers[i], 
			  quadrature_rule.points()[0], 1.0e-12);
    }
  }

  void testSimplePolygons()
  {
    //Create origin and unit vectors.
    Point_3 e0(0.0, 0.0, 0.0);

    Point_3 e1(1.0, 0.0, 0.0);
    Point_3 e2(0.0, 1.0, 0.0);
    Point_3 e3(0.0, 0.0, 1.0);

    Point_3 _e1(-1.0, 0.0, 0.0);
    Point_3 _e2(0.0, -1.0, 0.0);
    Point_3 _e3(0.0, 0.0, -1.0);

    //Create tetrahedrons with unit vectors and reference results.
    DoubleList reference_volumes;
    PointList reference_bary_centers;
    PolyhedronList reference_polyhedrons;

    //skew plane upper e3 plane
    add_test_polyhedron(e1,e2,e3,reference_polyhedrons);
    //todo find exact values
    reference_volumes.push_back(8.660254e-01);
    reference_bary_centers.push_back(Point(1.0/3.0,1.0/3.0,1.0/3.0));

    add_test_polyhedron(_e1,e2,e3,reference_polyhedrons);
    //todo find exact values
    reference_volumes.push_back(8.660254e-01);
    reference_bary_centers.push_back(Point(-1.0/3.0,1.0/3.0,1.0/3.0));

    add_test_polyhedron(_e1,_e2,e3,reference_polyhedrons);
    //todo find exact values
    reference_volumes.push_back(8.660254e-01);
    reference_bary_centers.push_back(Point(-1.0/3.0,-1.0/3.0,1.0/3.0));

    add_test_polyhedron(e1,_e2,e3,reference_polyhedrons);
    //todo find exact values
    reference_volumes.push_back(8.660254e-01);
    reference_bary_centers.push_back(Point(1.0/3.0,-1.0/3.0,1.0/3.0));

    //skew plane lower -e3 plane
    add_test_polyhedron(e1,e2,_e3,reference_polyhedrons);
    //todo find exact values
    reference_volumes.push_back(8.660254e-01);
    reference_bary_centers.push_back(Point(1.0/3.0,1.0/3.0,-1.0/3.0));

    add_test_polyhedron(_e1,e2,_e3,reference_polyhedrons);
    //todo find exact values
    reference_volumes.push_back(8.660254e-01);
    reference_bary_centers.push_back(Point(-1.0/3.0,1.0/3.0,-1.0/3.0));

    add_test_polyhedron(_e1,_e2,_e3,reference_polyhedrons);
    //todo find exact values
    reference_volumes.push_back(8.660254e-01);
    reference_bary_centers.push_back(Point(-1.0/3.0,-1.0/3.0,-1.0/3.0));

    add_test_polyhedron(e1,_e2,_e3,reference_polyhedrons);
    //todo find exact values
    reference_volumes.push_back(8.660254e-01);
    reference_bary_centers.push_back(Point(1.0/3.0,-1.0/3.0,-1.0/3.0));

    //e1-e2 plane
    add_test_polyhedron(e0,e1,e2,reference_polyhedrons);
    reference_volumes.push_back(1.0/2.0);
    reference_bary_centers.push_back(Point(1.0/3.0,1.0/3.0,0.0));

    add_test_polyhedron(e0,_e1,e2,reference_polyhedrons);
    reference_volumes.push_back(1.0/2.0);
    reference_bary_centers.push_back(Point(-1.0/3.0,1.0/3.0,0.0));

    add_test_polyhedron(e0,_e1,_e2,reference_polyhedrons);
    reference_volumes.push_back(1.0/2.0);
    reference_bary_centers.push_back(Point(-1.0/3.0,-1.0/3.0,0.0));

    add_test_polyhedron(e0,e1,_e2,reference_polyhedrons);
    reference_volumes.push_back(1.0/2.0);
    reference_bary_centers.push_back(Point(1.0/3.0,-1.0/3.0,0.0));

    //e1-e3 plane
    add_test_polyhedron(e0,e1,e3,reference_polyhedrons);
    reference_volumes.push_back(1.0/2.0);
    reference_bary_centers.push_back(Point(1.0/3.0,0.0,1.0/3.0));

    add_test_polyhedron(e0,_e1,e3,reference_polyhedrons);
    reference_volumes.push_back(1.0/2.0);
    reference_bary_centers.push_back(Point(-1.0/3.0,0.0,1.0/3.0));

    add_test_polyhedron(e0,_e1,_e3,reference_polyhedrons);
    reference_volumes.push_back(1.0/2.0);
    reference_bary_centers.push_back(Point(-1.0/3.0,0.0,-1.0/3.0));

    add_test_polyhedron(e0,e1,_e3,reference_polyhedrons);
    reference_volumes.push_back(1.0/2.0);
    reference_bary_centers.push_back(Point(1.0/3.0,0.0,-1.0/3.0));

    //e2-e3 plane
    add_test_polyhedron(e0,e2,e3,reference_polyhedrons);
    reference_volumes.push_back(1.0/2.0);
    reference_bary_centers.push_back(Point(0.0,1.0/3.0,1.0/3.0));

    add_test_polyhedron(e0,_e2,e3,reference_polyhedrons);
    reference_volumes.push_back(1.0/2.0);
    reference_bary_centers.push_back(Point(0.0,-1.0/3.0,1.0/3.0));

    add_test_polyhedron(e0,_e2,_e3,reference_polyhedrons);
    reference_volumes.push_back(1.0/2.0);
    reference_bary_centers.push_back(Point(0.0,-1.0/3.0,-1.0/3.0));

    add_test_polyhedron(e0,e2,_e3,reference_polyhedrons);
    reference_volumes.push_back(1.0/2.0);
    reference_bary_centers.push_back(Point(0.0,1.0/3.0,-1.0/3.0));

    //Test sum of polyhedrons
    IntList add_indices;
    add_indices.push_back(0);


    add_indices.push_back(1);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(2);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(3);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(4);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(5);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(6);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(7);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(8);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(9);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);


    add_indices.push_back(10);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(11);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(12);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(13);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(14);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    add_indices.push_back(15);
    add_disjoint_polyhedrons(add_indices, 
			     reference_polyhedrons, 
			     reference_volumes, 
			     reference_bary_centers);

    Nef_polyhedron_3 polyhedron = reference_polyhedrons[0];
    polyhedron.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(1, 1, 1)));
    reference_polyhedrons.push_back(polyhedron);
    reference_volumes.push_back(reference_volumes[0]); 
    reference_bary_centers.push_back(reference_bary_centers[0] + Point(1,1,1));  

    polyhedron = reference_polyhedrons[1];
    polyhedron.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(-1, 1, 1)));
    reference_polyhedrons.push_back(polyhedron);
    reference_volumes.push_back(reference_volumes[1]); 
    reference_bary_centers.push_back(reference_bary_centers[1] + Point(-1,1,1));  

    polyhedron = reference_polyhedrons[2];
    polyhedron.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(1, -1, 1)));
    reference_polyhedrons.push_back(polyhedron);
    reference_volumes.push_back(reference_volumes[2]); 
    reference_bary_centers.push_back(reference_bary_centers[2] + Point(1,-1,1));  

    polyhedron = reference_polyhedrons[3];
    polyhedron.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(-1, -1, 1)));
    reference_polyhedrons.push_back(polyhedron);
    reference_volumes.push_back(reference_volumes[3]); 
    reference_bary_centers.push_back(reference_bary_centers[3] + Point(-1,-1,1));  

    polyhedron = reference_polyhedrons[4];
    polyhedron.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(1, 1, -1)));
    reference_polyhedrons.push_back(polyhedron);
    reference_volumes.push_back(reference_volumes[4]); 
    reference_bary_centers.push_back(reference_bary_centers[4] + Point(1,1,-1));  

    polyhedron = reference_polyhedrons[5];
    polyhedron.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(-1, 1, -1)));
    reference_polyhedrons.push_back(polyhedron);
    reference_volumes.push_back(reference_volumes[5]); 
    reference_bary_centers.push_back(reference_bary_centers[5] + Point(-1,1,-1));  

    polyhedron = reference_polyhedrons[6];
    polyhedron.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(1, -1, -1)));
    reference_polyhedrons.push_back(polyhedron);
    reference_volumes.push_back(reference_volumes[6]); 
    reference_bary_centers.push_back(reference_bary_centers[6] + Point(1,-1,-1));  

    polyhedron = reference_polyhedrons[7];
    polyhedron.transform(Aff_transformation_3(CGAL::TRANSLATION, Vector_3(-1, -1, -1)));
    reference_polyhedrons.push_back(polyhedron);
    reference_volumes.push_back(reference_volumes[7]); 
    reference_bary_centers.push_back(reference_bary_centers[7] + Point(-1,-1,-1));  

    //Instantiate quadrature rule

    //Check volume and barycenter for polyhedrons
    for (dolfin::uint i = 0; i < reference_polyhedrons.size(); ++i)
    {
      BarycenterQuadrature quadrature_rule(reference_polyhedrons[i]);
      CPPUNIT_ASSERT_DOUBLES_EQUAL(reference_volumes[i],
				   quadrature_rule.weights()[0], 1.0e-5);
      almost_equal_points(reference_bary_centers[i], 
			  quadrature_rule.points()[0], 1.0e-5);
    }
  }

  void testComplexPolyhedrons()
  {
  }

  void testComplexPolygons()
  {
  }

};

#else

//No tests if CGAL is not available.

class BaryCenter : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(BaryCenter);
  CPPUNIT_TEST_SUITE_END();
};

#endif

CPPUNIT_TEST_SUITE_REGISTRATION(BaryCenter);

int main()
{
  DOLFIN_TEST;
}

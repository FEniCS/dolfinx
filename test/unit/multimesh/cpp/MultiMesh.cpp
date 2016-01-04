// Copyright (C) 2014 August Johansson and Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2014-03-10
// Last changed: 2015-10-25
//
// Unit tests for MultiMesh

#include <dolfin.h>
#include <dolfin/common/unittest.h>
//FIXME August
#include <dolfin/geometry/dolfin_simplex_tools.h>
#include <dolfin/geometry/predicates.h>

#define MULTIMESH_DEBUG_OUTPUT 1

using namespace dolfin;

class MultiMeshes : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MultiMeshes);
  //CPPUNIT_TEST(test_multiple_meshes_with_rotation);
  //CPPUNIT_TEST(test_multiple_meshes_with_dynamic_rotation);
  //CPPUNIT_TEST(test_exclusion_inclusion);
  //CPPUNIT_TEST(test_exclusion_inclusion_small_angle);
  //CPPUNIT_TEST(test_multiple_meshes_quadrature);
  CPPUNIT_TEST(test_multiple_meshes_interface_quadrature);
  //CPPUNIT_TEST(test_mass_matrix);
  CPPUNIT_TEST_SUITE_END();

public:

  // void test_mass_matrix()
  // {
  //   UnitSquareMesh mesh_0(1, 1);
  //   UnitSquareMesh mesh_1(2, 2);
  //   MultiMesh multimesh;
  //   multimesh.add(mesh_0);
  //   multimesh.add(mesh_1);
  //   multimesh.build();
  // }

  //------------------------------------------------------------------------------
  void rotate(double x, double y, double cx, double cy, double w,
	      double& xr, double& yr)
  {
    // std::cout << "rotate:\n"
    // 	      << "\t"
    // 	      << "plot("<<x<<','<<y<<",'b.');plot("<<cx<<','<<cy<<",'o');";

    const double v = w*DOLFIN_PI/180.;
    const double dx = x-cx;
    const double dy = y-cy;
    xr = cx + dx*cos(v) - dy*sin(v);
    yr = cy + dx*sin(v) + dy*cos(v);
    //std::cout << "plot("<<xr<<','<<yr<<",'r.');"<<std::endl;
  }

  bool rotation_inside(double x,double y, double cx, double cy, double w,
		       double& xr, double& yr)
  {
    rotate(x,y,cx,cy,w, xr,yr);
    if (xr>0 and xr<1 and yr>0 and yr<1) return true;
    else return false;
  }

  struct MeshData
  {
    MeshData() {}
    MeshData(double xa, double ya, double xb, double yb, std::size_t i, std::size_t j, double w, double s)
      : x0(xa), x1(xb), y0(ya), y1(yb), v(w), cx(0.5*(xa+xb)), cy(0.5*(ya+yb)), speed(s),
	m(i), n(j) {}
    // void update(double w)
    // {
    //   const double cx = (x0+x1) / 2;
    //   const double cy = (y0+y1) / 2;
    //   const double a=x0, b=x1, c=y0, d=y1;
    //   rotate(a,c,cx,cy,w, x0,y0);
    //   rotate(b,d,cx,cy,w, x1,y1);
    //   v=w;
    // }
    friend std::ostream& operator<<(std::ostream &out, const MeshData& md)
    {
      out<<md.x0<<' '<<md.y0<<' '<<md.x1<<' '<<md.y1<<' '<<md.m<<' '<<md.n<<' '<<md.v<<' '<<md.speed;
      return out;
    }
    double x0, y0, x1, y1, v, cx, cy, speed;
    std::size_t m, n;
  };

  void test_multiple_meshes_with_rotation()
  {
    set_log_level(DBG);

    dolfin::seed(0);

    const double h = 0.5;
    UnitSquareMesh background_mesh((int)std::round(1./h),
				   (int)std::round(1./h));
    MultiMesh multimesh;
    multimesh.add(background_mesh);

    const std::size_t Nmeshes = 8;

    std::size_t i = 0;
    while (i < Nmeshes)
    {
      double x0 = dolfin::rand();
      double x1 = dolfin::rand();
      if (x0 > x1) std::swap(x0, x1);
      double y0 = dolfin::rand();
      double y1 = dolfin::rand();
      if (y0 > y1) std::swap(y0, y1);
      const double v = dolfin::rand()*90; // initial rotation
      const double speed = dolfin::rand()-0.5; // initial speed
      const double cx = (x0+x1) / 2;
      const double cy = (y0+y1) / 2;
      double xr, yr;
      rotate(x0, y0, cx, cy, v, xr, yr);
      if (xr > 0 and xr < 1 and yr > 0 and yr < 1)
      {
	rotate(x0, y1, cx, cy, v, xr, yr);
	if (xr > 0 and xr < 1 and yr > 0 and yr < 1)
	{
	  rotate(x1, y0, cx, cy, v, xr, yr);
	  if (xr > 0 and xr < 1 and yr > 0 and yr < 1)
	  {
	    rotate(x1, y1, cx, cy, v, xr, yr);
	    if (xr > 0 and xr < 1 and yr > 0 and yr < 1)
	    {
              std::shared_ptr<Mesh> mesh(new RectangleMesh(Point(x0, y0), Point(x1, y1),
                                                           std::max((int)std::round((x1-x0)/h), 1),
                                                           std::max((int)std::round((y1-y0)/h), 1)));
              mesh->rotate(v);

              multimesh.add(mesh);

	      i++;
	    }
	  }
	}
      }
    }

    multimesh.build();

#ifdef MULTIMESH_DEBUG_OUTPUT
    tools::dolfin_write_medit_triangles("multimesh",multimesh);
    std::cout << multimesh.plot_matplotlib() << std::endl;
#endif

    // Exact volume is known
    const double exact_volume = 1;
    const double volume = compute_volume(multimesh, exact_volume);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);

  }

  //------------------------------------------------------------------------------
  bool find_rotated_mesh(const MeshData& md,
			 double v,
			 MeshData& mdnew)
  {
    //std::cout << "try rotate " << md << " with angle " << md.speed*v << std::endl;

    double x0,y0,x1,y1,b,c,d,e,f;
    const double w = md.v+md.speed*v;
    if (rotation_inside(md.x0, md.y0, md.cx, md.cy, w, x0,y0) and
	rotation_inside(md.x0, md.y1, md.cx, md.cy, w, c,d) and
	rotation_inside(md.x1, md.y0, md.cx, md.cy, w, e,f) and
	rotation_inside(md.x1, md.y1, md.cx, md.cy, w, x1,y1))
    {
      mdnew = MeshData(md.x0,md.y0, md.x1,md.y1, md.m, md.n, w, md.speed);
      //std::cout << "rotated mesh found"<<std::endl;
      return true;
    }
    else return false;
  }

  void test_multiple_meshes_with_dynamic_rotation()
  {
    exactinit();

    //set_log_level(DEBUG);
    dolfin::seed(0);

    double max_error = -1;
    const double h = 0.1;
    UnitSquareMesh background_mesh((int)std::round(1./h),
				   (int)std::round(1./h));

    // Create data for Nmeshes
    const std::size_t Nmeshes = 20;
    std::vector<MeshData> md(Nmeshes);
    std::size_t i = 0;
    //std::cout << "Create initial meshes\n";
    while (i < Nmeshes)
    {
      const double x0 = dolfin::rand();
      const double x1 = dolfin::rand();
      const double y0 = dolfin::rand();
      const double y1 = dolfin::rand();
      const double v = dolfin::rand()*90; // initial rotation
      const double speed = dolfin::rand()-0.5;
      MeshData mdold(x0,y0, x1,y1,
		     std::max((int)std::round((x1-x0)/h), 1),
		     std::max((int)std::round((y1-y0)/h), 1),
		     v, speed);
      bool mesh_ok = find_rotated_mesh(mdold, v, md[i]);
      if (mesh_ok)
      {
	//std::cout << i << ' ' << md[i] << std::endl;
	i++;
      }
      //else{std::cout << "try again\n"; }
    }

    // Create rotations
    const std::size_t Nangles = 5*180;
    const double angle_step = 5*180. / Nangles;
    std::size_t cnt = 0;

    for (std::size_t j = 0; j < Nangles; ++j)
    {
      MultiMesh multimesh;
      multimesh.add(background_mesh);

      const double v = j*angle_step;
      std::cout << "Rotate angle="<<v << " (step="<< j << ")"<<std::endl;

      for (std::size_t i = 0; i < Nmeshes; ++i)
      {
	MeshData mdnew;
	bool mesh_ok = find_rotated_mesh(md[i], angle_step, mdnew);
	if (!mesh_ok) // try -v
	{
	  //std::cout << "flip speed\n";
	  md[i].speed *= -1;
	  mesh_ok = find_rotated_mesh(md[i], angle_step, mdnew);
	}

	if (!mesh_ok)
	{
	  std::cout << i<<' '<<md[i] <<std::endl;
	  PPause;
	} // should work

	if (mesh_ok)
	  md[i] = mdnew;
      }

      RectangleMesh mesh_0(Point(md[0].x0, md[0].y0), Point(md[0].x1, md[0].y1), md[0].m, md[0].n);
      mesh_0.rotate(md[0].v);
      multimesh.add(mesh_0);
      RectangleMesh mesh_1(Point(md[1].x0, md[1].y0), Point(md[1].x1, md[1].y1),  md[1].m, md[1].n);
      mesh_1.rotate(md[1].v);
      multimesh.add(mesh_1);
      RectangleMesh mesh_2(Point(md[2].x0, md[2].y0), Point(md[2].x1, md[2].y1),  md[2].m, md[2].n);
      mesh_2.rotate(md[2].v);
      multimesh.add(mesh_2);
      RectangleMesh mesh_3(Point(md[3].x0, md[3].y0), Point(md[3].x1, md[3].y1),  md[3].m, md[3].n);
      mesh_3.rotate(md[3].v);
      multimesh.add(mesh_3);
      RectangleMesh mesh_4(Point(md[4].x0, md[4].y0), Point(md[4].x1, md[4].y1),  md[4].m, md[4].n);
      mesh_4.rotate(md[4].v);
      multimesh.add(mesh_4);
      RectangleMesh mesh_5(Point(md[5].x0, md[5].y0), Point(md[5].x1, md[5].y1),  md[5].m, md[5].n);
      mesh_5.rotate(md[5].v);
      multimesh.add(mesh_5);
      RectangleMesh mesh_6(Point(md[6].x0, md[6].y0), Point(md[6].x1, md[6].y1),  md[6].m, md[6].n);
      mesh_6.rotate(md[6].v);
      multimesh.add(mesh_6);
      RectangleMesh mesh_7(Point(md[7].x0, md[7].y0), Point(md[7].x1, md[7].y1),  md[7].m, md[7].n);
      mesh_7.rotate(md[7].v);
      multimesh.add(mesh_7);
      // RectangleMesh mesh_8(Point(md[8].x0, md[8].y0), Point(md[8].x1, md[8].y1),  md[8].m, md[8].n);
      // mesh_8.rotate(md[8].v);
      // multimesh.add(mesh_8);
      // RectangleMesh mesh_9(Point(md[9].x0, md[9].y0), Point(md[9].x1, md[9].y1),  md[9].m, md[9].n);
      // mesh_9.rotate(md[9].v);
      // multimesh.add(mesh_9);
      // RectangleMesh mesh_10(Point(md[10].x0, md[10].y0), Point(md[10].x1, md[10].y1),  md[10].m, md[10].n);
      // mesh_10.rotate(md[10].v);
      // multimesh.add(mesh_10);
      // RectangleMesh mesh_11(Point(md[11].x0, md[11].y0), Point(md[11].x1, md[11].y1),  md[11].m, md[11].n);
      // mesh_11.rotate(md[11].v);
      // multimesh.add(mesh_11);
      // RectangleMesh mesh_12(Point(md[12].x0, md[12].y0), Point(md[12].x1, md[12].y1),  md[12].m, md[12].n);
      // mesh_12.rotate(md[12].v);
      // multimesh.add(mesh_12);
      // RectangleMesh mesh_13(Point(md[13].x0, md[13].y0), Point(md[13].x1, md[13].y1),  md[13].m, md[13].n);
      // mesh_13.rotate(md[13].v);
      // multimesh.add(mesh_13);
      // RectangleMesh mesh_14(Point(md[14].x0, md[14].y0), Point(md[14].x1, md[14].y1),  md[14].m, md[14].n);
      // mesh_14.rotate(md[14].v);
      // multimesh.add(mesh_14);
      // RectangleMesh mesh_15(Point(md[15].x0, md[15].y0), Point(md[15].x1, md[15].y1),  md[15].m, md[15].n);
      // mesh_15.rotate(md[15].v);
      // multimesh.add(mesh_15);
      // RectangleMesh mesh_16(Point(md[16].x0, md[16].y0), Point(md[16].x1, md[16].y1),  md[16].m, md[16].n);
      // mesh_16.rotate(md[16].v);
      // multimesh.add(mesh_16);
      // RectangleMesh mesh_17(Point(md[17].x0, md[17].y0), Point(md[17].x1, md[17].y1),  md[17].m, md[17].n);
      // mesh_17.rotate(md[17].v);
      // multimesh.add(mesh_17);
      // RectangleMesh mesh_18(Point(md[18].x0, md[18].y0), Point(md[18].x1, md[18].y1),  md[18].m, md[18].n);
      // mesh_18.rotate(md[18].v);
      // multimesh.add(mesh_18);
      // RectangleMesh mesh_19(Point(md[19].x0, md[19].y0), Point(md[19].x1, md[19].y1),  md[19].m, md[19].n);
      // mesh_19.rotate(md[19].v);
      // multimesh.add(mesh_19);

      multimesh.build();

      if (j%10==0)
	tools::dolfin_write_medit_triangles("after_multimesh",multimesh, cnt++);

      // Exact volume is known
      const double exact_volume = 1;
      const double volume = compute_volume(multimesh, exact_volume);
      const double e = std::abs(volume - exact_volume);
      max_error = std::max(e, max_error);
      std::cout << std::setprecision(15)
		<< "volume = " << volume << '\n'
		<< "current error = " << e << '\n'
		<< "max_error = " << max_error << '\n';

      //CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
    }

  }

  //------------------------------------------------------------------------------
  void test_exclusion_inclusion()
  {
    set_log_level(DBG);

    const double v = 1e-16;

    UnitSquareMesh mesh_0(1, 1);
    RectangleMesh mesh_1(Point(0.2, 0.2), Point(0.8, 0.8), 1, 1);
    mesh_1.rotate(v, 2);

    RectangleMesh mesh_2(Point(0.3, 0.3), Point(0.7, 0.7), 1, 1);
    //mesh_2.rotate(8.002805e-01, 2);
    //mesh_2.rotate(1.418863e-01, 2);
    mesh_2.rotate(2*v, 2);

    // RectangleMesh mesh_3(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // //mesh_3.rotate(1.418863e-01, 2);
    // //mesh_3.rotate(4.217613e-01, 2);
    // mesh_3.rotate(0.002, 2);

    // RectangleMesh mesh_4(0.200000, 0.200000, 0.800000, 0.800000, 2, 1);
    // //mesh_4.rotate(4.217613e-01, 2);
    // //mesh_4.rotate(8.002805e-01, 2);
    // mesh_4.rotate(0.003, 2);

    // // RectangleMesh mesh_5(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // // mesh_5.rotate(0.004, 2);
    // // RectangleMesh mesh_6(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // // mesh_6.rotate(7.922073e-01, 2);
    // // RectangleMesh mesh_7(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // // mesh_7.rotate(9.594924e-01, 2);
    // // RectangleMesh mesh_8(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // // mesh_8.rotate(6.557407e-01, 2);
    // // RectangleMesh mesh_9(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // // mesh_9.rotate(3.571168e-02, 2);
    // // RectangleMesh mesh_10(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // // mesh_10.rotate(8.491293e-01, 2);
    // // RectangleMesh mesh_11(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // // mesh_11.rotate(9.339932e-01, 2);
    // // RectangleMesh mesh_12(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // // mesh_12.rotate(6.787352e-01, 2);
    // // RectangleMesh mesh_13(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // // mesh_13.rotate(7.577401e-01, 2);
    // // RectangleMesh mesh_14(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // // mesh_14.rotate(7.431325e-01, 2);
    // // RectangleMesh mesh_15(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // // mesh_15.rotate(3.922270e-01, 2);

    MultiMesh multimesh;
    multimesh.add(mesh_0);
    multimesh.add(mesh_1);
    multimesh.add(mesh_2);
    // multimesh.add(mesh_3);
    // multimesh.add(mesh_4);
    // multimesh.add(mesh_5);
    // multimesh.add(mesh_6);
    // multimesh.add(mesh_7);
    // multimesh.add(mesh_8);
    // multimesh.add(mesh_9);
    // multimesh.add(mesh_10);
    // multimesh.add(mesh_11);
    // multimesh.add(mesh_12);
    // multimesh.add(mesh_13);
    // multimesh.add(mesh_14);
    // multimesh.add(mesh_15);
    multimesh.build();



    // UnitSquareMesh mesh_0(1, 1);
    // RectangleMesh mesh_1(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_1.rotate(5, 2);
    // RectangleMesh mesh_2(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_2.rotate(10, 2);
    // RectangleMesh mesh_3(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_3.rotate(15, 2);
    // RectangleMesh mesh_4(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_4.rotate(20, 2);
    // RectangleMesh mesh_5(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_5.rotate(25, 2);
    // RectangleMesh mesh_6(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_6.rotate(30, 2);
    // RectangleMesh mesh_7(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_7.rotate(35, 2);
    // RectangleMesh mesh_8(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_8.rotate(40, 2);
    // RectangleMesh mesh_9(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_9.rotate(45, 2);
    // RectangleMesh mesh_10(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_10.rotate(50, 2);

    // MultiMesh multimesh;
    // multimesh.add(mesh_0);
    // multimesh.add(mesh_1);
    // multimesh.add(mesh_2);
    // multimesh.add(mesh_3);
    // multimesh.add(mesh_4);
    // multimesh.add(mesh_5);
    // multimesh.add(mesh_6);
    // multimesh.add(mesh_7);
    // multimesh.add(mesh_8);
    // multimesh.add(mesh_9);
    // multimesh.add(mesh_10);
    // multimesh.build();


    // UnitSquareMesh mesh_0(1, 1);
    // RectangleMesh mesh_1(0.2, 0.2, 0.8, 0.8, 1, 1);
    // RectangleMesh mesh_2(0.3, 0.3, 0.7, 0.7, 1, 1);

    // // Build the multimesh
    // MultiMesh multimesh;
    // multimesh.add(mesh_0);
    // multimesh.add(mesh_1);
    // multimesh.add(mesh_2);
    // multimesh.build();


    // UnitSquareMesh mesh_0(4, 4);
    // RectangleMesh mesh_1(0.2, 0.2, 0.8, 0.8, 4, 4);
    // mesh_1.rotate(10, 2);
    // RectangleMesh mesh_2(0.2, 0.2, 0.8, 0.8, 4, 4);
    // mesh_2.rotate(20, 2);
    // RectangleMesh mesh_3(0.2, 0.2, 0.8, 0.8, 4, 4);
    // mesh_3.rotate(30, 2);
    // RectangleMesh mesh_4(0.2, 0.2, 0.8, 0.8, 4, 4);
    // mesh_4.rotate(40, 2);
    // RectangleMesh mesh_5(0.2, 0.2, 0.8, 0.8, 4, 4);
    // mesh_5.rotate(50, 2);
    // RectangleMesh mesh_6(0.2, 0.2, 0.8, 0.8, 4, 4);
    // mesh_6.rotate(60, 2);
    // RectangleMesh mesh_7(0.2, 0.2, 0.8, 0.8, 4, 4);
    // mesh_7.rotate(70, 2);
    // RectangleMesh mesh_8(0.2, 0.2, 0.8, 0.8, 4, 4);
    // mesh_8.rotate(80, 2);
    // RectangleMesh mesh_9(0.2, 0.2, 0.8, 0.8, 4, 4);
    // mesh_9.rotate(90, 2);

    // // Build the multimesh
    // MultiMesh multimesh;
    // multimesh.add(mesh_0);
    // multimesh.add(mesh_1);
    // multimesh.add(mesh_2);
    // multimesh.add(mesh_3);
    // multimesh.add(mesh_4);
    // multimesh.add(mesh_5);
    // multimesh.add(mesh_6);
    // //multimesh.add(mesh_7);
    // // multimesh.add(mesh_8);
    // // multimesh.add(mesh_9);
    // multimesh.build();



    // UnitSquareMesh mesh_0(1, 1);
    // RectangleMesh mesh_1(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_1.rotate(5, 2);
    // RectangleMesh mesh_2(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_2.rotate(10, 2);
    // RectangleMesh mesh_3(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_3.rotate(15, 2);
    // RectangleMesh mesh_4(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_4.rotate(20, 2);
    // RectangleMesh mesh_5(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_5.rotate(25, 2);
    // RectangleMesh mesh_6(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_6.rotate(30, 2);
    // RectangleMesh mesh_7(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_7.rotate(35, 2);
    // RectangleMesh mesh_8(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_8.rotate(40, 2);
    // RectangleMesh mesh_9(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_9.rotate(45, 2);
    // RectangleMesh mesh_10(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    // mesh_10.rotate(50, 2);

    // MultiMesh multimesh;
    // multimesh.add(mesh_0);
    // multimesh.add(mesh_1);
    // multimesh.add(mesh_2);
    // multimesh.add(mesh_3);
    // multimesh.add(mesh_4);
    // multimesh.add(mesh_5);
    // multimesh.add(mesh_6);
    // multimesh.add(mesh_7);
    // multimesh.add(mesh_8);
    // multimesh.add(mesh_9);
    // multimesh.add(mesh_10);
    // multimesh.build();


    // UnitSquareMesh mesh_0(1, 1);
    // RectangleMesh mesh_1(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // mesh_1.rotate(6, 2);
    // RectangleMesh mesh_2(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // mesh_2.rotate(12, 2);
    // RectangleMesh mesh_3(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // mesh_3.rotate(18, 2);
    // RectangleMesh mesh_4(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // mesh_4.rotate(24, 2);
    // RectangleMesh mesh_5(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // mesh_5.rotate(30, 2);
    // RectangleMesh mesh_6(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // mesh_6.rotate(36, 2);
    // RectangleMesh mesh_7(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // mesh_7.rotate(42, 2);
    // RectangleMesh mesh_8(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // mesh_8.rotate(48, 2);
    // RectangleMesh mesh_9(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // mesh_9.rotate(54, 2);
    // RectangleMesh mesh_10(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // mesh_10.rotate(60, 2);
    // RectangleMesh mesh_11(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // mesh_11.rotate(66, 2);
    // RectangleMesh mesh_12(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // mesh_12.rotate(72, 2);
    // RectangleMesh mesh_13(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // mesh_13.rotate(78, 2);
    // RectangleMesh mesh_14(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // mesh_14.rotate(84, 2);
    // RectangleMesh mesh_15(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // mesh_15.rotate(90, 2);

    // MultiMesh multimesh;
    // multimesh.add(mesh_0);
    // multimesh.add(mesh_1);
    // multimesh.add(mesh_2);
    // multimesh.add(mesh_3);
    // multimesh.add(mesh_4);
    // multimesh.add(mesh_5);
    // multimesh.add(mesh_6);
    // multimesh.add(mesh_7);
    // multimesh.add(mesh_8);
    // multimesh.add(mesh_9);
    // multimesh.add(mesh_10);
    // multimesh.add(mesh_11);
    // // multimesh.add(mesh_12);
    // // multimesh.add(mesh_13);
    // // multimesh.add(mesh_14);
    // //multimesh.add(mesh_15);
    // multimesh.build();



    // UnitSquareMesh mesh_0(1, 1);
    // RectangleMesh mesh_1(0.2, 0.2, 0.8, 0.8, 1, 1);
    // RectangleMesh mesh_2(0.2, 0.2, 0.8, 0.8, 1, 1);
    // mesh_2.rotate(1, 2);
    // RectangleMesh mesh_3(0.2, 0.2, 0.8, 0.8, 1, 1);
    // mesh_3.rotate(2, 2);
    // RectangleMesh mesh_4(0.2, 0.2, 0.8, 0.8, 1, 1);
    // mesh_4.rotate(3, 2);
    // RectangleMesh mesh_5(0.2, 0.2, 0.8, 0.8, 1, 1);
    // mesh_5.rotate(4, 2);
    // RectangleMesh mesh_6(0.2, 0.2, 0.8, 0.8, 1, 1);
    // mesh_6.rotate(5, 2);
    // RectangleMesh mesh_7(0.2, 0.2, 0.8, 0.8, 1, 1);
    // mesh_7.rotate(6, 2);
    // RectangleMesh mesh_8(0.2, 0.2, 0.8, 0.8, 1, 1);
    // mesh_8.rotate(7, 2);
    // RectangleMesh mesh_9(0.2, 0.2, 0.8, 0.8, 1, 1);
    // mesh_9.rotate(8, 2);
    // RectangleMesh mesh_10(0.2, 0.2, 0.8, 0.8, 1, 1);
    // mesh_10.rotate(9, 2);
    // RectangleMesh mesh_11(0.2, 0.2, 0.8, 0.8, 1, 1);
    // mesh_11.rotate(10, 2);
    // RectangleMesh mesh_12(0.2, 0.2, 0.8, 0.8, 1, 1);
    // mesh_12.rotate(11, 2);
    // RectangleMesh mesh_13(0.2, 0.2, 0.8, 0.8, 1, 1);
    // mesh_13.rotate(12, 2);
    // RectangleMesh mesh_14(0.2, 0.2, 0.8, 0.8, 1, 1);
    // mesh_14.rotate(13, 2);
    // RectangleMesh mesh_15(0.2, 0.2, 0.8, 0.8, 1, 1);
    // mesh_15.rotate(14, 2);

    // MultiMesh multimesh;
    // multimesh.add(mesh_0);
    // multimesh.add(mesh_1);
    // multimesh.add(mesh_2);
    // multimesh.add(mesh_3);
    // multimesh.add(mesh_4);
    // // multimesh.add(mesh_5);
    // // multimesh.add(mesh_6);
    // // multimesh.add(mesh_7);
    // // multimesh.add(mesh_8);
    // // multimesh.add(mesh_9);
    // // multimesh.add(mesh_10);
    // // multimesh.add(mesh_11);
    // // multimesh.add(mesh_12);
    // // multimesh.add(mesh_13);
    // // multimesh.add(mesh_14);
    // // multimesh.add(mesh_15);
    // multimesh.build();

    // UnitSquareMesh mesh_0(1, 1);
    // RectangleMesh mesh_1(0.814724, 0.126987, 0.905792, 0.913376, 1, 1);
    // RectangleMesh mesh_2(0.957507, 0.157613, 0.964889, 0.970593, 1, 1);
    // RectangleMesh mesh_3(0.421761, 0.792207, 0.915736, 0.959492, 1, 1);
    // RectangleMesh mesh_4(0.186873, 0.445586, 0.489764, 0.646313, 1, 1);
    // RectangleMesh mesh_5(0.709365, 0.276025, 0.754687, 0.679703, 1, 1);
    // RectangleMesh mesh_6(0.162182, 0.311215, 0.794285, 0.528533, 1, 1);
    // RectangleMesh mesh_7(0.165649, 0.262971, 0.601982, 0.654079, 1, 1);
    // RectangleMesh mesh_8(0.228977, 0.152378, 0.913337, 0.825817, 1, 1);
    // RectangleMesh mesh_9(0.538342, 0.078176, 0.996135, 0.442678, 1, 1);
    // RectangleMesh mesh_10(0.106653, 0.004634, 0.961898, 0.774910, 1, 1);
    // RectangleMesh mesh_11(0.817303, 0.084436, 0.868695, 0.399783, 1, 1);
    // RectangleMesh mesh_12(0.259870, 0.431414, 0.800068, 0.910648, 1, 1);
    // RectangleMesh mesh_13(0.096455, 0.942051, 0.131973, 0.956135, 1, 1);
    // RectangleMesh mesh_14(0.296676, 0.424167, 0.318778, 0.507858, 1, 1);
    // RectangleMesh mesh_15(0.098712, 0.335357, 0.261871, 0.679728, 1, 1);
    // RectangleMesh mesh_16(0.136553, 0.106762, 0.721227, 0.653757, 1, 1);
    // RectangleMesh mesh_17(0.494174, 0.715037, 0.779052, 0.903721, 1, 1);
    // RectangleMesh mesh_18(0.059619, 0.042431, 0.681972, 0.071445, 1, 1);
    // RectangleMesh mesh_19(0.432392, 0.083470, 0.825314, 0.133171, 1, 1);
    // RectangleMesh mesh_20(0.269119, 0.547871, 0.422836, 0.942737, 1, 1);
    // RectangleMesh mesh_21(0.417744, 0.301455, 0.983052, 0.701099, 1, 1);
    // RectangleMesh mesh_22(0.190433, 0.460726, 0.368917, 0.981638, 1, 1);
    // RectangleMesh mesh_23(0.068806, 0.530864, 0.319600, 0.654446, 1, 1);
    // RectangleMesh mesh_24(0.407619, 0.718359, 0.819981, 0.968649, 1, 1);
    // RectangleMesh mesh_25(0.153657, 0.440085, 0.281005, 0.527143, 1, 1);
    // // RectangleMesh mesh_26(0.457424, 0.518052, 0.875372, 0.943623, 1, 1);
    // // RectangleMesh mesh_27(0.637709, 0.240707, 0.957694, 0.676122, 1, 1);
    // // RectangleMesh mesh_28(0.341125, 0.191745, 0.607389, 0.738427, 1, 1);
    // // RectangleMesh mesh_29(0.242850, 0.269062, 0.917424, 0.765500, 1, 1);
    // // RectangleMesh mesh_30(0.188662, 0.091113, 0.287498, 0.576209, 1, 1);
    // // RectangleMesh mesh_31(0.647618, 0.635787, 0.679017, 0.945174, 1, 1);
    // // RectangleMesh mesh_32(0.479463, 0.544716, 0.639317, 0.647311, 1, 1);
    // // RectangleMesh mesh_33(0.543886, 0.522495, 0.721047, 0.993705, 1, 1);
    // // RectangleMesh mesh_34(0.404580, 0.365816, 0.448373, 0.763505, 1, 1);
    // // RectangleMesh mesh_35(0.627896, 0.932854, 0.771980, 0.972741, 1, 1);
    // // RectangleMesh mesh_36(0.044454, 0.242785, 0.754933, 0.442402, 1, 1);
    // // RectangleMesh mesh_37(0.327565, 0.438645, 0.671264, 0.833501, 1, 1);
    // // RectangleMesh mesh_38(0.199863, 0.748706, 0.406955, 0.825584, 1, 1);
    // // RectangleMesh mesh_39(0.582791, 0.879014, 0.815397, 0.988912, 1, 1);
    // // RectangleMesh mesh_40(0.000522, 0.612566, 0.865439, 0.989950, 1, 1);
    // // RectangleMesh mesh_41(0.498094, 0.574661, 0.900852, 0.845178, 1, 1);
    // // RectangleMesh mesh_42(0.083483, 0.660945, 0.625960, 0.729752, 1, 1);
    // // RectangleMesh mesh_43(0.552291, 0.031991, 0.629883, 0.614713, 1, 1);
    // // RectangleMesh mesh_44(0.123084, 0.146515, 0.205494, 0.189072, 1, 1);
    // // RectangleMesh mesh_45(0.042652, 0.281867, 0.635198, 0.538597, 1, 1);
    // // RectangleMesh mesh_46(0.123932, 0.852998, 0.490357, 0.873927, 1, 1);
    // // RectangleMesh mesh_47(0.105709, 0.166460, 0.142041, 0.620959, 1, 1);
    // // RectangleMesh mesh_48(0.467068, 0.025228, 0.648198, 0.842207, 1, 1);
    // // RectangleMesh mesh_49(0.559033, 0.347879, 0.854100, 0.446027, 1, 1);
    // // RectangleMesh mesh_50(0.706917, 0.287849, 0.999492, 0.414523, 1, 1);

    // MultiMesh multimesh;
    // multimesh.add(mesh_0);
    // multimesh.add(mesh_1);
    // multimesh.add(mesh_2);
    // multimesh.add(mesh_3);
    // multimesh.add(mesh_4);
    // multimesh.add(mesh_5);
    // multimesh.add(mesh_6);
    // multimesh.add(mesh_7);
    // multimesh.add(mesh_8);
    // multimesh.add(mesh_9);
    // multimesh.add(mesh_10);
    // multimesh.add(mesh_11);
    // multimesh.add(mesh_12);
    // multimesh.add(mesh_13);
    // multimesh.add(mesh_14);
    // multimesh.add(mesh_15);
    // multimesh.add(mesh_16);
    // multimesh.add(mesh_17);
    // multimesh.add(mesh_18);
    // multimesh.add(mesh_19);
    // multimesh.add(mesh_20);
    // multimesh.add(mesh_21);
    // multimesh.add(mesh_22);
    // multimesh.add(mesh_23);
    // multimesh.add(mesh_24);
    // multimesh.add(mesh_25);
    // // multimesh.add(mesh_26);
    // // multimesh.add(mesh_27);
    // // multimesh.add(mesh_28);
    // // multimesh.add(mesh_29);
    // // multimesh.add(mesh_30);
    // // multimesh.add(mesh_31);
    // // multimesh.add(mesh_32);
    // // multimesh.add(mesh_33);
    // // multimesh.add(mesh_34);
    // // multimesh.add(mesh_35);
    // // multimesh.add(mesh_36);
    // // multimesh.add(mesh_37);
    // // multimesh.add(mesh_38);
    // // multimesh.add(mesh_39);
    // // multimesh.add(mesh_40);
    // // multimesh.add(mesh_41);
    // // multimesh.add(mesh_42);
    // // multimesh.add(mesh_43);
    // // multimesh.add(mesh_44);
    // // multimesh.add(mesh_45);
    // // multimesh.add(mesh_46);
    // // multimesh.add(mesh_47);
    // // multimesh.add(mesh_48);
    // // multimesh.add(mesh_49);
    // // multimesh.add(mesh_50);
    // multimesh.build();


    // UnitSquareMesh mesh_0(1, 1);
    // RectangleMesh mesh_1(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // RectangleMesh mesh_2(0.200001, 0.200001, 0.800001, 0.800001, 1, 1);
    // RectangleMesh mesh_3(0.200002, 0.200002, 0.800002, 0.800002, 1, 1);
    // RectangleMesh mesh_4(0.200003, 0.200003, 0.800003, 0.800003, 1, 1);
    // RectangleMesh mesh_5(0.200004, 0.200004, 0.800004, 0.800004, 1, 1);
    // RectangleMesh mesh_6(0.200005, 0.200005, 0.800005, 0.800005, 1, 1);
    // RectangleMesh mesh_7(0.200006, 0.200006, 0.800006, 0.800006, 1, 1);
    // RectangleMesh mesh_8(0.200007, 0.200007, 0.800007, 0.800007, 1, 1);
    // RectangleMesh mesh_9(0.200008, 0.200008, 0.800008, 0.800008, 1, 1);
    // RectangleMesh mesh_10(0.200009, 0.200009, 0.800009, 0.800009, 1, 1);
    // RectangleMesh mesh_11(0.200010, 0.200010, 0.800010, 0.800010, 1, 1);
    // RectangleMesh mesh_12(0.200011, 0.200011, 0.800011, 0.800011, 1, 1);
    // RectangleMesh mesh_13(0.200012, 0.200012, 0.800012, 0.800012, 1, 1);
    // RectangleMesh mesh_14(0.200013, 0.200013, 0.800013, 0.800013, 1, 1);
    // RectangleMesh mesh_15(0.200014, 0.200014, 0.800014, 0.800014, 1, 1);

    // MultiMesh multimesh;
    // multimesh.add(mesh_0);
    // multimesh.add(mesh_1);
    // multimesh.add(mesh_2);
    // multimesh.add(mesh_3);
    // multimesh.add(mesh_4);
    // multimesh.add(mesh_5);
    // multimesh.add(mesh_6);
    // multimesh.add(mesh_7);
    // multimesh.add(mesh_8);
    // multimesh.add(mesh_9);
    // multimesh.add(mesh_10);
    // multimesh.add(mesh_11);
    // multimesh.add(mesh_12);
    // multimesh.add(mesh_13);
    // multimesh.add(mesh_14);
    // multimesh.add(mesh_15);
    // multimesh.build();

    // Parameters p;
    // p("multimesh")["quadrature_order"] = 6;

    // MultiMeshFunctionSpace W;
    // W.parameters("multimesh")["quadrature_order"] = 6;


    tools::dolfin_write_medit_triangles("multimesh",multimesh);
    // for (std::size_t part = 0; part < multimesh.num_parts(); part++)
    // {
    //   tools::write_vtu_hack("mesh",*multimesh.part(part),part);
    //   // std::stringstream ss; ss << part;
    //   // File file("mesh"+ss.str()+".vtk");
    //   // file << *multimesh.part(part);
    // }


    // Exact volume is known
    const double exact_volume = 1;
    const double volume = compute_volume(multimesh, exact_volume);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
  }


  // void test_multiple_meshes_quadrature()
  // {
  //   set_log_level(DEBUG);

  //   // Create multimesh from three triangle meshes of the unit square

  //   // // Many meshes, but not more than three overlap => this works
  //   // UnitCubeMesh mesh_0(11, 12, 13);
  //   // BoxMesh mesh_1(0.1, 0.1, 0.1,    0.9, 0.9, 0.9,     13, 11, 12);
  //   // BoxMesh mesh_2(0.2, 0.2, 0.2,    0.95, 0.95, 0.8,   11, 13, 11);
  //   // BoxMesh mesh_3(0.94, 0.01, 0.01,  0.98, 0.99, 0.99, 1, 11, 11);
  //   // BoxMesh mesh_4(0.01, 0.01, 0.01, 0.02, 0.02, 0.02,  1, 1, 1);

  //   // Completely nested 2D: can't do no more than three meshes
  //   UnitSquareMesh mesh_0(1, 1);
  //   RectangleMesh mesh_1(0.1, 0.1, 0.9, 0.9, 1, 1);
  //   RectangleMesh mesh_2(0.2, 0.2, 0.8, 0.8, 1, 1);
  //   // RectangleMesh mesh_3(0.3, 0.3, 0.7, 0.7, 1, 1);
  //   // RectangleMesh mesh_4(0.4, 0.4, 0.6, 0.6, 1, 1);

  //   // // Completely nested 3D: can't do no more than three meshes
  //   // UnitCubeMesh mesh_0(2, 3, 4);
  //   // BoxMesh mesh_1(0.1, 0.1, 0.1,    0.9, 0.9, 0.9,   4, 3, 2);
  //   // BoxMesh mesh_2(0.2, 0.2, 0.2,    0.8, 0.8, 0.8,   3, 4, 3);
  //   // BoxMesh mesh_3(0.8, 0.01, 0.01,  0.9, 0.99, 0.99,  4, 2, 3);
  //   // BoxMesh mesh_4(0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 1, 1, 1);


  //   // Build the multimesh
  //   MultiMesh multimesh;
  //   multimesh.add(mesh_0);
  //   multimesh.add(mesh_1);
  //   multimesh.add(mesh_2);
  //   // multimesh.add(mesh_3);
  //   // multimesh.add(mesh_4);
  //   multimesh.build();

  //   // Exact volume is known
  //   const double exact_volume = 1;
  //   double volume = 0;

  //   // Sum contribution from all parts
  //   std::cout << "Sum contributions\n";
  //   for (std::size_t part = 0; part < multimesh.num_parts(); part++)
  //   {
  //     std::cout << "% part " << part;
  //     double part_volume = 0;

  //     // Uncut cell volume given by function volume
  //     const auto uncut_cells = multimesh.uncut_cells(part);
  //     for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
  //     {
  //       const Cell cell(*multimesh.part(part), *it);
  //       volume += cell.volume();
  //       part_volume += cell.volume();
  //     }

  //     std::cout << "\t uncut volume "<< part_volume<<' ';

  //     // Cut cell volume given by quadrature rule
  //     const auto& cut_cells = multimesh.cut_cells(part);
  //     for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
  //     {
  //       const auto& qr = multimesh.quadrature_rule_cut_cell(part, *it);
  //       for (std::size_t i = 0; i < qr.second.size(); ++i)
  //       {
  //         volume += qr.second[i];
  //         part_volume += qr.second[i];
  //       }
  //     }
  //     std::cout << "\ttotal volume " << part_volume<< std::endl;
  //   }

  //   std::cout<<std::setprecision(13) << "exact volume " << exact_volume<<'\n'
  //             << "volume " << volume<<std::endl;
  //   CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
  // }

  void test_multiple_meshes_interface_quadrature()
  {

    // // Test triangles on top of each other
    // for (std::size_t N = 1; N < 20; ++N)
    // {
    //   for (std::size_t k = 0; k < 80; ++k)
    // 	std::cout << '-';
    //   std::cout << "\nN = " << N << std::endl;
    //   MultiMesh multimesh;
    //   MeshEditor me;
    //   std::vector<Mesh> meshes(N);
    //   double exact_area = 0;
    //   for (std::size_t i = 0; i < N; ++i)
    //   {
    // 	me.open(meshes[i], 2, 2);
    // 	me.init_vertices(3);
    // 	me.init_cells(1);
    // 	const double a = 0.01;
    // 	const Point p0(a*i, a*i);
    // 	const Point p1(1-2*a*i, a*i);
    // 	const Point p2(a*i, 1-2*a*i);
    // 	me.add_vertex(0, p0);
    // 	me.add_vertex(1, p1);
    // 	me.add_vertex(2, p2);
    // 	me.add_cell(0, 0, 1, 2);
    // 	me.close();
    // 	multimesh.add(meshes[i]);
    // 	const std::vector<double>& x=meshes[i].coordinates();
    // 	if (i > 0)
    // 	  exact_area += p0.distance(p1) + p0.distance(p2) + p1.distance(p2);
    //   }
    //   multimesh.build();
    //   tools::dolfin_write_medit_triangles("multimesh",multimesh, N);

    //   const double area = compute_interface_area(multimesh, exact_area);
    //   const double e = std::abs(area - exact_area);
    //   std::cout << std::setprecision(15)
    // 		<< "N = " << N << '\n'
    // 		<< "area = " << area << '\n'
    // 		<< "error = " << e << '\n';
    //   //CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_area, area, DOLFIN_EPS_LARGE);
    // }

    // {
    //   MeshEditor me;
    //   Mesh mesh;
    //   me.open(mesh, 2, 2);
    //   me.init_vertices(3);
    //   me.init_cells(1);
    //   const Point p0(0., 0.);
    //   const Point p1(1., 0.);
    //   const Point p2(0., 1.);
    //   me.add_vertex(0, p0);
    //   me.add_vertex(1, p1);
    //   me.add_vertex(2, p2);
    //   me.add_cell(0, 0, 1, 2);
    //   me.close();

    //   MeshEditor meline;
    //   Mesh meshline;
    //   meline.open(meshline, 1, 2);
    //   meline.init_vertices(2);
    //   meline.init_cells(1);
    //   const Point a(0.5, 0.2);
    //   const Point b(0.5, -0.5);
    //   meline.add_vertex(0, a);
    //   meline.add_vertex(1, b);
    //   const std::vector<std::size_t> v = {{0, 1}};
    //   meline.add_cell(0, v);
    //   meline.close();

    //   Cell tri(mesh, 0);
    //   Cell line(meshline, 0);

    //   std::cout << tools::drawtriangle(tri)<<tools::drawtriangle(line)<<std::endl;

    //   const std::vector<double> intersection = IntersectionTriangulation::triangulate_intersection(tri, line);
    //   for (const auto t: intersection)
    // 	std::cout << t <<' ';
    //   std::cout << std::endl;

    //   std::cout << tools::drawtriangle(intersection,"'r'")<<std::endl;

    //   exit(0);
    // }



    // Test squares in diagonal on background unit square
    const std::size_t m = 5, n = 5;
    const double h = 0.4;
    const double s = 0.5;
    if (h >= s) { std::cout << "h must be less than s\n"; exit(1); }
    UnitSquareMesh usm(m, n);
    MultiMesh mm;
    mm.add(usm);
    std::vector<Mesh> meshes;
    double exact_area = 4*s;
    std::size_t N = 1;
    while (N*h+s < 1)
    {
      std::cout << "rectangle mesh points [" << N*h << "," << N*h+s << "] x [" << N*h << "," << N*h+s << "]" << std::endl;
      std::shared_ptr<Mesh> rm(new RectangleMesh(Point(N*h, N*h), Point(N*h+s, N*h+s), m, n));
      mm.add(rm);
      if (N > 1)
	exact_area += 2*s + 2*h;
      N++;
    }
    mm.build();
    tools::dolfin_write_medit_triangles("multimesh", mm, N);

    const double area = compute_interface_area(mm, exact_area);
    const double error = std::abs(area - exact_area);
    std::cout << std::setprecision(15)
	      << "N = " << N << '\n'
	      << "area = " << area << '\n'
	      << "error = " << error << '\n';
    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_area, area, DOLFIN_EPS_LARGE);
  }


  //------------------------------------------------------------------------------
  void test_exclusion_inclusion_small_angle()
  {
    //set_log_level(DEBUG);

    exactinit();

    std::stringstream ss;
    ss << "angle_output_90.txt";
    std::ofstream file(ss.str());
    if (!file.good()) { std::cout << ss.str() << " not ok" << std::endl; exit(0); }
    file.precision(15);

    std::vector<double> angles;
    double v = 100;
    while (v > 1e-17)
    {
      angles.push_back(v);
      v /= 10;
    }
    //angles.push_back(1e-7);
    // for (std::size_t i = 1; i < 90; ++i)
    // 	angles.push_back(i);

    double max_error = -1;

    for (const auto v: angles)
    {
      std::cout << "--------------------------------------\n"
		<< "try v = " << v << std::endl;
      for (std::size_t m = 1; m <= 100; ++m)
	for (std::size_t n = 1; n <= 100; ++n)
	{
	  UnitSquareMesh mesh_0(m, n);
	  RectangleMesh mesh_1(Point(0.2, 0.2), Point(0.8, 0.8), m, n);
	  mesh_1.rotate(v, 2);

	  MultiMesh multimesh;
	  multimesh.add(mesh_0);
	  multimesh.add(mesh_1);
	  multimesh.build();

	  tools::dolfin_write_medit_triangles("multimesh",multimesh);

	  const double exact_volume = 1;
	  const double volume = compute_volume(multimesh, exact_volume);
	  const double e = std::abs(volume - exact_volume);
	  max_error = std::max(e, max_error);
	  std::cout << std::setprecision(15)
		    << "v = " << v << '\n'
		    << "m = " << m << '\n'
		    << "n = " << n << '\n'
		    << "volume = " << volume << '\n'
		    << "error = " << e << '\n'
		    << "max error = " << max_error << '\n';
	  file << v <<' '<< m<<' '<<n<<' '<<volume << ' '<<e << std::endl;

	  //CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
	}
    }
  }


  double compute_volume(const MultiMesh& multimesh,
			double exact_volume) const
  {
    std::cout << '\n';

    double volume = 0;
    std::vector<double> all_volumes;

    std::ofstream file("quadrature.txt");
    if (!file.good()) { std::cout << "file not good\n"; exit(0); }
    file.precision(20);

    // Sum contribution from all parts
    std::cout << "Sum contributions\n";
    for (std::size_t part = 0; part < multimesh.num_parts(); part++)
    {
      std::cout << "% part " << part;
      double part_volume = 0;
      std::vector<double> status(multimesh.part(part)->num_cells(), 0);

      // Uncut cell volume given by function volume
      const auto uncut_cells = multimesh.uncut_cells(part);
      for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
      {
        const Cell cell(*multimesh.part(part), *it);
        volume += cell.volume();
	//std::cout << std::setprecision(20) << cell.volume() <<'\n';
        part_volume += cell.volume();
	status[*it] = 1;
	//file << "0 0 "<< cell.volume() << '\n';
      }

      std::cout << "\t uncut volume "<< part_volume << ' ';

      // Cut cell volume given by quadrature rule
      const auto& cut_cells = multimesh.cut_cells(part);
      for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
      {
        const auto& qr = multimesh.quadrature_rule_cut_cell(part, *it);
        for (std::size_t i = 0; i < qr.second.size(); ++i)
        {
	  file << qr.first[2*i]<<' '<<qr.first[2*i+1]<<' '<<qr.second[i]<<'\n';
          volume += qr.second[i];
          part_volume += qr.second[i];
	  //std::cout << qr.first[2*i]<<' '<<qr.first[2*i+1]<<'\n';
        }
	status[*it] = 2;
      }
      std::cout << "\ttotal volume " << part_volume << std::endl;

      all_volumes.push_back(part_volume);

      tools::dolfin_write_medit_triangles("status",*multimesh.part(part),part,&status);
    }
    file.close();

    return volume;
  }

  double compute_interface_area(const MultiMesh& multimesh,
				double exact_area) const
  {
    std::cout << '\n';

    double area = 0;
    std::vector<double> all_areas;

    std::ofstream file("quadrature.txt");
    if (!file.good()) { std::cout << "file not good\n"; exit(0); }
    file.precision(20);

    // Sum contribution from all parts
    std::cout << "Sum contributions\n";
    for (std::size_t part = 0; part < multimesh.num_parts(); part++)
    {
      std::cout << "% part " << part << std::endl;
      double part_area = 0;
      const auto& quadrature_rules = multimesh.quadrature_rule_interface(part);

      // // Uncut cell area given by function area
      // const auto uncut_cells = multimesh.uncut_cells(part);
      // for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
      // {
      //   const Cell cell(*multimesh.part(part), *it);
      //   area += cell.area();
      // 	//std::cout << std::setprecision(20) << cell.area() <<'\n';
      //   part_area += cell.area();
      // 	status[*it] = 1;
      // 	//file << "0 0 "<< cell.area() << '\n';
      // }

      // std::cout << "\t uncut area "<< part_area << ' ';


      // Get collision map
      const auto& cmap = multimesh.collision_map_cut_cells(part);
      for (auto it = cmap.begin(); it != cmap.end(); ++it)
      {
	const unsigned int cut_cell_index = it->first;
	const auto& cutting_cells = it->second;

	// Iterate over cutting cells
	for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
	{
	  // Get quadrature rule for interface part defined by
	  // intersection of the cut and cutting cells
	  const std::size_t k = jt - cutting_cells.begin();
	  // std::cout << cut_cell_index << ' ' << k <<' ' << std::flush
	  // 	    << quadrature_rules.size() << ' '
	  // 	    << quadrature_rules.at(cut_cell_index).size() << "   " << std::flush;
	  dolfin_assert(k < quadrature_rules.at(cut_cell_index).size());
	  const auto& qr = quadrature_rules.at(cut_cell_index)[k];
	  std::stringstream ss;
	  for (std::size_t i = 0; i < qr.second.size(); ++i)
	  {
	    file << qr.first[2*i]<<' '<<qr.first[2*i+1]<<' '<<qr.second[i]<<'\n';
	    //std::cout << qr.second[i]<<' ';
	    area += qr.second[i];
	    part_area += qr.second[i];
	    //std::cout << qr.first[2*i]<<' '<<qr.first[2*i+1]<<'\n';
	  }
	  //std::cout << std::endl;
	}
      }
      std::cout << "total area " << part_area << std::endl;
      all_areas.push_back(part_area);
    }
    file.close();

    return area;
  }

};

int main()
{
  // Test not workin in parallel
  if (dolfin::MPI::size(MPI_COMM_WORLD) > 1)
  {
    info("Skipping unit test in parallel.");
    info("OK");
    return 0;
  }

  CPPUNIT_TEST_SUITE_REGISTRATION(MultiMeshes);
  DOLFIN_TEST;
}

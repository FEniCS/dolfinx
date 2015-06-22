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
// Last changed: 2015-06-08
//
// Unit tests for MultiMesh

#include <dolfin.h>
#include <dolfin/common/unittest.h>
//FIXME August
#include <dolfin_simplex_tools.h>
#include <dolfin/geometry/predicates.h>

using namespace dolfin;

class MultiMeshes : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MultiMeshes);
  //CPPUNIT_TEST(test_multiple_meshes_with_rotation);
  CPPUNIT_TEST(test_multiple_meshes_with_dynamic_rotation);
  //CPPUNIT_TEST(test_exclusion_inclusion);
  //CPPUNIT_TEST(test_exclusion_inclusion_small_angle);
  //CPPUNIT_TEST(test_multiple_meshes_quadrature);
  //CPPUNIT_TEST(test_multiple_meshes_interface_quadrature);
  CPPUNIT_TEST_SUITE_END();

public:


  //------------------------------------------------------------------------------
  double rotate(double x, double y, double cx, double cy, double w,
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
    // std::cout << "plot("<<xr<<','<<yr<<",'r.');"<<std::endl;
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
    MeshData(double xa, double ya, double xb, double yb, std::size_t i, std::size_t j, double w)
      : x0(xa), x1(xb), y0(ya), y1(yb), m(i), n(j), v(w), cx(0.5*(xa+xb)), cy(0.5*(ya+yb)) {}
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
      out<<md.x0<<' '<<md.y0<<' '<<md.x1<<' '<<md.y1<<' '<<md.m<<' '<<md.n<<' '<<md.v;
      return out;
    }
    double x0, y0, x1, y1, v, cx, cy;
    std::size_t m, n;
  };

  void test_multiple_meshes_with_rotation()
  {
    set_log_level(DEBUG);

    dolfin::seed(0);

    const double h = 0.1;
    UnitSquareMesh background_mesh((int)std::round(1./h),
				   (int)std::round(1./h));
    MultiMesh multimesh;
    multimesh.add(background_mesh);

    const std::size_t Nmeshes = 100;

    std::vector<MeshData> md(Nmeshes);

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
	      md[i] = MeshData(x0, y0, x1, y1,
			       std::max((int)std::round((x1-x0)/h), 1),
			       std::max((int)std::round((y1-y0)/h), 1),
			       v);
	      std::cout << i << ' ' << md[i] << std::endl;
	      i++;
	    }
	  }
	}
      }
    }


    RectangleMesh mesh_0(md[0].x0, md[0].y0, md[0].x1, md[0].y1, md[0].m, md[0].n);
    mesh_0.rotate(md[0].v);
    RectangleMesh mesh_1(md[1].x0, md[1].y0, md[1].x1, md[1].y1, md[1].m, md[1].n);
    mesh_1.rotate(md[1].v);
    RectangleMesh mesh_2(md[2].x0, md[2].y0, md[2].x1, md[2].y1, md[2].m, md[2].n);
    mesh_2.rotate(md[2].v);
    RectangleMesh mesh_3(md[3].x0, md[3].y0, md[3].x1, md[3].y1, md[3].m, md[3].n);
    mesh_3.rotate(md[3].v);
    RectangleMesh mesh_4(md[4].x0, md[4].y0, md[4].x1, md[4].y1, md[4].m, md[4].n);
    mesh_4.rotate(md[4].v);
    RectangleMesh mesh_5(md[5].x0, md[5].y0, md[5].x1, md[5].y1, md[5].m, md[5].n);
    mesh_5.rotate(md[5].v);
    RectangleMesh mesh_6(md[6].x0, md[6].y0, md[6].x1, md[6].y1, md[6].m, md[6].n);
    mesh_6.rotate(md[6].v);
    RectangleMesh mesh_7(md[7].x0, md[7].y0, md[7].x1, md[7].y1, md[7].m, md[7].n);
    mesh_7.rotate(md[7].v);
    RectangleMesh mesh_8(md[8].x0, md[8].y0, md[8].x1, md[8].y1, md[8].m, md[8].n);
    mesh_8.rotate(md[8].v);
    RectangleMesh mesh_9(md[9].x0, md[9].y0, md[9].x1, md[9].y1, md[9].m, md[9].n);
    mesh_9.rotate(md[9].v);
    RectangleMesh mesh_10(md[10].x0, md[10].y0, md[10].x1, md[10].y1, md[10].m, md[10].n);
    mesh_10.rotate(md[10].v);
    RectangleMesh mesh_11(md[11].x0, md[11].y0, md[11].x1, md[11].y1, md[11].m, md[11].n);
    mesh_11.rotate(md[11].v);
    RectangleMesh mesh_12(md[12].x0, md[12].y0, md[12].x1, md[12].y1, md[12].m, md[12].n);
    mesh_12.rotate(md[12].v);
    RectangleMesh mesh_13(md[13].x0, md[13].y0, md[13].x1, md[13].y1, md[13].m, md[13].n);
    mesh_13.rotate(md[13].v);
    RectangleMesh mesh_14(md[14].x0, md[14].y0, md[14].x1, md[14].y1, md[14].m, md[14].n);
    mesh_14.rotate(md[14].v);
    RectangleMesh mesh_15(md[15].x0, md[15].y0, md[15].x1, md[15].y1, md[15].m, md[15].n);
    mesh_15.rotate(md[15].v);
    RectangleMesh mesh_16(md[16].x0, md[16].y0, md[16].x1, md[16].y1, md[16].m, md[16].n);
    mesh_16.rotate(md[16].v);
    RectangleMesh mesh_17(md[17].x0, md[17].y0, md[17].x1, md[17].y1, md[17].m, md[17].n);
    mesh_17.rotate(md[17].v);
    RectangleMesh mesh_18(md[18].x0, md[18].y0, md[18].x1, md[18].y1, md[18].m, md[18].n);
    mesh_18.rotate(md[18].v);
    RectangleMesh mesh_19(md[19].x0, md[19].y0, md[19].x1, md[19].y1, md[19].m, md[19].n);
    mesh_19.rotate(md[19].v);
    RectangleMesh mesh_20(md[20].x0, md[20].y0, md[20].x1, md[20].y1, md[20].m, md[20].n);
    mesh_20.rotate(md[20].v);
    RectangleMesh mesh_21(md[21].x0, md[21].y0, md[21].x1, md[21].y1, md[21].m, md[21].n);
    mesh_21.rotate(md[21].v);
    RectangleMesh mesh_22(md[22].x0, md[22].y0, md[22].x1, md[22].y1, md[22].m, md[22].n);
    mesh_22.rotate(md[22].v);
    RectangleMesh mesh_23(md[23].x0, md[23].y0, md[23].x1, md[23].y1, md[23].m, md[23].n);
    mesh_23.rotate(md[23].v);
    RectangleMesh mesh_24(md[24].x0, md[24].y0, md[24].x1, md[24].y1, md[24].m, md[24].n);
    mesh_24.rotate(md[24].v);
    RectangleMesh mesh_25(md[25].x0, md[25].y0, md[25].x1, md[25].y1, md[25].m, md[25].n);
    mesh_25.rotate(md[25].v);
    RectangleMesh mesh_26(md[26].x0, md[26].y0, md[26].x1, md[26].y1, md[26].m, md[26].n);
    mesh_26.rotate(md[26].v);
    RectangleMesh mesh_27(md[27].x0, md[27].y0, md[27].x1, md[27].y1, md[27].m, md[27].n);
    mesh_27.rotate(md[27].v);
    RectangleMesh mesh_28(md[28].x0, md[28].y0, md[28].x1, md[28].y1, md[28].m, md[28].n);
    mesh_28.rotate(md[28].v);
    RectangleMesh mesh_29(md[29].x0, md[29].y0, md[29].x1, md[29].y1, md[29].m, md[29].n);
    mesh_29.rotate(md[29].v);
    RectangleMesh mesh_30(md[30].x0, md[30].y0, md[30].x1, md[30].y1, md[30].m, md[30].n);
    mesh_30.rotate(md[30].v);
    RectangleMesh mesh_31(md[31].x0, md[31].y0, md[31].x1, md[31].y1, md[31].m, md[31].n);
    mesh_31.rotate(md[31].v);
    RectangleMesh mesh_32(md[32].x0, md[32].y0, md[32].x1, md[32].y1, md[32].m, md[32].n);
    mesh_32.rotate(md[32].v);
    RectangleMesh mesh_33(md[33].x0, md[33].y0, md[33].x1, md[33].y1, md[33].m, md[33].n);
    mesh_33.rotate(md[33].v);
    RectangleMesh mesh_34(md[34].x0, md[34].y0, md[34].x1, md[34].y1, md[34].m, md[34].n);
    mesh_34.rotate(md[34].v);
    RectangleMesh mesh_35(md[35].x0, md[35].y0, md[35].x1, md[35].y1, md[35].m, md[35].n);
    mesh_35.rotate(md[35].v);
    RectangleMesh mesh_36(md[36].x0, md[36].y0, md[36].x1, md[36].y1, md[36].m, md[36].n);
    mesh_36.rotate(md[36].v);
    RectangleMesh mesh_37(md[37].x0, md[37].y0, md[37].x1, md[37].y1, md[37].m, md[37].n);
    mesh_37.rotate(md[37].v);
    RectangleMesh mesh_38(md[38].x0, md[38].y0, md[38].x1, md[38].y1, md[38].m, md[38].n);
    mesh_38.rotate(md[38].v);
    RectangleMesh mesh_39(md[39].x0, md[39].y0, md[39].x1, md[39].y1, md[39].m, md[39].n);
    mesh_39.rotate(md[39].v);
    RectangleMesh mesh_40(md[40].x0, md[40].y0, md[40].x1, md[40].y1, md[40].m, md[40].n);
    mesh_40.rotate(md[40].v);
    RectangleMesh mesh_41(md[41].x0, md[41].y0, md[41].x1, md[41].y1, md[41].m, md[41].n);
    mesh_41.rotate(md[41].v);
    RectangleMesh mesh_42(md[42].x0, md[42].y0, md[42].x1, md[42].y1, md[42].m, md[42].n);
    mesh_42.rotate(md[42].v);
    RectangleMesh mesh_43(md[43].x0, md[43].y0, md[43].x1, md[43].y1, md[43].m, md[43].n);
    mesh_43.rotate(md[43].v);
    RectangleMesh mesh_44(md[44].x0, md[44].y0, md[44].x1, md[44].y1, md[44].m, md[44].n);
    mesh_44.rotate(md[44].v);
    RectangleMesh mesh_45(md[45].x0, md[45].y0, md[45].x1, md[45].y1, md[45].m, md[45].n);
    mesh_45.rotate(md[45].v);
    RectangleMesh mesh_46(md[46].x0, md[46].y0, md[46].x1, md[46].y1, md[46].m, md[46].n);
    mesh_46.rotate(md[46].v);
    RectangleMesh mesh_47(md[47].x0, md[47].y0, md[47].x1, md[47].y1, md[47].m, md[47].n);
    mesh_47.rotate(md[47].v);
    RectangleMesh mesh_48(md[48].x0, md[48].y0, md[48].x1, md[48].y1, md[48].m, md[48].n);
    mesh_48.rotate(md[48].v);
    RectangleMesh mesh_49(md[49].x0, md[49].y0, md[49].x1, md[49].y1, md[49].m, md[49].n);
    mesh_49.rotate(md[49].v);
    RectangleMesh mesh_50(md[50].x0, md[50].y0, md[50].x1, md[50].y1, md[50].m, md[50].n);
    mesh_50.rotate(md[50].v);
    RectangleMesh mesh_51(md[51].x0, md[51].y0, md[51].x1, md[51].y1, md[51].m, md[51].n);
    mesh_51.rotate(md[51].v);
    RectangleMesh mesh_52(md[52].x0, md[52].y0, md[52].x1, md[52].y1, md[52].m, md[52].n);
    mesh_52.rotate(md[52].v);
    RectangleMesh mesh_53(md[53].x0, md[53].y0, md[53].x1, md[53].y1, md[53].m, md[53].n);
    mesh_53.rotate(md[53].v);
    RectangleMesh mesh_54(md[54].x0, md[54].y0, md[54].x1, md[54].y1, md[54].m, md[54].n);
    mesh_54.rotate(md[54].v);
    RectangleMesh mesh_55(md[55].x0, md[55].y0, md[55].x1, md[55].y1, md[55].m, md[55].n);
    mesh_55.rotate(md[55].v);
    RectangleMesh mesh_56(md[56].x0, md[56].y0, md[56].x1, md[56].y1, md[56].m, md[56].n);
    mesh_56.rotate(md[56].v);
    RectangleMesh mesh_57(md[57].x0, md[57].y0, md[57].x1, md[57].y1, md[57].m, md[57].n);
    mesh_57.rotate(md[57].v);
    RectangleMesh mesh_58(md[58].x0, md[58].y0, md[58].x1, md[58].y1, md[58].m, md[58].n);
    mesh_58.rotate(md[58].v);
    RectangleMesh mesh_59(md[59].x0, md[59].y0, md[59].x1, md[59].y1, md[59].m, md[59].n);
    mesh_59.rotate(md[59].v);
    RectangleMesh mesh_60(md[60].x0, md[60].y0, md[60].x1, md[60].y1, md[60].m, md[60].n);
    mesh_60.rotate(md[60].v);
    RectangleMesh mesh_61(md[61].x0, md[61].y0, md[61].x1, md[61].y1, md[61].m, md[61].n);
    mesh_61.rotate(md[61].v);
    RectangleMesh mesh_62(md[62].x0, md[62].y0, md[62].x1, md[62].y1, md[62].m, md[62].n);
    mesh_62.rotate(md[62].v);
    RectangleMesh mesh_63(md[63].x0, md[63].y0, md[63].x1, md[63].y1, md[63].m, md[63].n);
    mesh_63.rotate(md[63].v);
    RectangleMesh mesh_64(md[64].x0, md[64].y0, md[64].x1, md[64].y1, md[64].m, md[64].n);
    mesh_64.rotate(md[64].v);
    RectangleMesh mesh_65(md[65].x0, md[65].y0, md[65].x1, md[65].y1, md[65].m, md[65].n);
    mesh_65.rotate(md[65].v);
    RectangleMesh mesh_66(md[66].x0, md[66].y0, md[66].x1, md[66].y1, md[66].m, md[66].n);
    mesh_66.rotate(md[66].v);
    RectangleMesh mesh_67(md[67].x0, md[67].y0, md[67].x1, md[67].y1, md[67].m, md[67].n);
    mesh_67.rotate(md[67].v);
    RectangleMesh mesh_68(md[68].x0, md[68].y0, md[68].x1, md[68].y1, md[68].m, md[68].n);
    mesh_68.rotate(md[68].v);
    RectangleMesh mesh_69(md[69].x0, md[69].y0, md[69].x1, md[69].y1, md[69].m, md[69].n);
    mesh_69.rotate(md[69].v);
    RectangleMesh mesh_70(md[70].x0, md[70].y0, md[70].x1, md[70].y1, md[70].m, md[70].n);
    mesh_70.rotate(md[70].v);
    RectangleMesh mesh_71(md[71].x0, md[71].y0, md[71].x1, md[71].y1, md[71].m, md[71].n);
    mesh_71.rotate(md[71].v);
    RectangleMesh mesh_72(md[72].x0, md[72].y0, md[72].x1, md[72].y1, md[72].m, md[72].n);
    mesh_72.rotate(md[72].v);
    RectangleMesh mesh_73(md[73].x0, md[73].y0, md[73].x1, md[73].y1, md[73].m, md[73].n);
    mesh_73.rotate(md[73].v);
    RectangleMesh mesh_74(md[74].x0, md[74].y0, md[74].x1, md[74].y1, md[74].m, md[74].n);
    mesh_74.rotate(md[74].v);
    RectangleMesh mesh_75(md[75].x0, md[75].y0, md[75].x1, md[75].y1, md[75].m, md[75].n);
    mesh_75.rotate(md[75].v);
    RectangleMesh mesh_76(md[76].x0, md[76].y0, md[76].x1, md[76].y1, md[76].m, md[76].n);
    mesh_76.rotate(md[76].v);
    RectangleMesh mesh_77(md[77].x0, md[77].y0, md[77].x1, md[77].y1, md[77].m, md[77].n);
    mesh_77.rotate(md[77].v);
    RectangleMesh mesh_78(md[78].x0, md[78].y0, md[78].x1, md[78].y1, md[78].m, md[78].n);
    mesh_78.rotate(md[78].v);
    RectangleMesh mesh_79(md[79].x0, md[79].y0, md[79].x1, md[79].y1, md[79].m, md[79].n);
    mesh_79.rotate(md[79].v);
    RectangleMesh mesh_80(md[80].x0, md[80].y0, md[80].x1, md[80].y1, md[80].m, md[80].n);
    mesh_80.rotate(md[80].v);
    RectangleMesh mesh_81(md[81].x0, md[81].y0, md[81].x1, md[81].y1, md[81].m, md[81].n);
    mesh_81.rotate(md[81].v);
    RectangleMesh mesh_82(md[82].x0, md[82].y0, md[82].x1, md[82].y1, md[82].m, md[82].n);
    mesh_82.rotate(md[82].v);
    RectangleMesh mesh_83(md[83].x0, md[83].y0, md[83].x1, md[83].y1, md[83].m, md[83].n);
    mesh_83.rotate(md[83].v);
    RectangleMesh mesh_84(md[84].x0, md[84].y0, md[84].x1, md[84].y1, md[84].m, md[84].n);
    mesh_84.rotate(md[84].v);
    RectangleMesh mesh_85(md[85].x0, md[85].y0, md[85].x1, md[85].y1, md[85].m, md[85].n);
    mesh_85.rotate(md[85].v);
    RectangleMesh mesh_86(md[86].x0, md[86].y0, md[86].x1, md[86].y1, md[86].m, md[86].n);
    mesh_86.rotate(md[86].v);
    RectangleMesh mesh_87(md[87].x0, md[87].y0, md[87].x1, md[87].y1, md[87].m, md[87].n);
    mesh_87.rotate(md[87].v);
    RectangleMesh mesh_88(md[88].x0, md[88].y0, md[88].x1, md[88].y1, md[88].m, md[88].n);
    mesh_88.rotate(md[88].v);
    RectangleMesh mesh_89(md[89].x0, md[89].y0, md[89].x1, md[89].y1, md[89].m, md[89].n);
    mesh_89.rotate(md[89].v);
    RectangleMesh mesh_90(md[90].x0, md[90].y0, md[90].x1, md[90].y1, md[90].m, md[90].n);
    mesh_90.rotate(md[90].v);
    RectangleMesh mesh_91(md[91].x0, md[91].y0, md[91].x1, md[91].y1, md[91].m, md[91].n);
    mesh_91.rotate(md[91].v);
    RectangleMesh mesh_92(md[92].x0, md[92].y0, md[92].x1, md[92].y1, md[92].m, md[92].n);
    mesh_92.rotate(md[92].v);
    RectangleMesh mesh_93(md[93].x0, md[93].y0, md[93].x1, md[93].y1, md[93].m, md[93].n);
    mesh_93.rotate(md[93].v);
    RectangleMesh mesh_94(md[94].x0, md[94].y0, md[94].x1, md[94].y1, md[94].m, md[94].n);
    mesh_94.rotate(md[94].v);
    RectangleMesh mesh_95(md[95].x0, md[95].y0, md[95].x1, md[95].y1, md[95].m, md[95].n);
    mesh_95.rotate(md[95].v);
    RectangleMesh mesh_96(md[96].x0, md[96].y0, md[96].x1, md[96].y1, md[96].m, md[96].n);
    mesh_96.rotate(md[96].v);
    RectangleMesh mesh_97(md[97].x0, md[97].y0, md[97].x1, md[97].y1, md[97].m, md[97].n);
    mesh_97.rotate(md[97].v);
    RectangleMesh mesh_98(md[98].x0, md[98].y0, md[98].x1, md[98].y1, md[98].m, md[98].n);
    mesh_98.rotate(md[98].v);
    RectangleMesh mesh_99(md[99].x0, md[99].y0, md[99].x1, md[99].y1, md[99].m, md[99].n);
    mesh_99.rotate(md[99].v);

    multimesh.add(mesh_0);
    multimesh.add(mesh_1);
    multimesh.add(mesh_2);
    multimesh.add(mesh_3);
    multimesh.add(mesh_4);
    multimesh.add(mesh_5);
    multimesh.add(mesh_6);
    multimesh.add(mesh_7);
    multimesh.add(mesh_8);
    multimesh.add(mesh_9);
    multimesh.add(mesh_10);
    multimesh.add(mesh_11);
    multimesh.add(mesh_12);
    multimesh.add(mesh_13);
    multimesh.add(mesh_14);
    multimesh.add(mesh_15);
    multimesh.add(mesh_16);
    multimesh.add(mesh_17);
    multimesh.add(mesh_18);
    multimesh.add(mesh_19);
    multimesh.add(mesh_20);
    // multimesh.add(mesh_21);
    // multimesh.add(mesh_22);
    // multimesh.add(mesh_23);
    // multimesh.add(mesh_24);
    // multimesh.add(mesh_25);
    // multimesh.add(mesh_26);
    // multimesh.add(mesh_27);
    // multimesh.add(mesh_28);
    // multimesh.add(mesh_29);
    // multimesh.add(mesh_30);
    // multimesh.add(mesh_31);
    // multimesh.add(mesh_32);
    // multimesh.add(mesh_33);
    // multimesh.add(mesh_34);
    // multimesh.add(mesh_35);
    // multimesh.add(mesh_36);
    // multimesh.add(mesh_37);
    // multimesh.add(mesh_38);
    // multimesh.add(mesh_39);
    // multimesh.add(mesh_40);
    // multimesh.add(mesh_41);
    // multimesh.add(mesh_42);
    // multimesh.add(mesh_43);
    // multimesh.add(mesh_44);
    // multimesh.add(mesh_45);
    // multimesh.add(mesh_46);
    // multimesh.add(mesh_47);
    // multimesh.add(mesh_48);
    // multimesh.add(mesh_49);
    // multimesh.add(mesh_50);
    // multimesh.add(mesh_51);
    // multimesh.add(mesh_52);
    // multimesh.add(mesh_53);
    // multimesh.add(mesh_54);
    // multimesh.add(mesh_55);
    // multimesh.add(mesh_56);
    // multimesh.add(mesh_57);
    // multimesh.add(mesh_58);
    // multimesh.add(mesh_59);
    // multimesh.add(mesh_60);
    // multimesh.add(mesh_61);
    // multimesh.add(mesh_62);
    // multimesh.add(mesh_63);
    // multimesh.add(mesh_64);
    // multimesh.add(mesh_65);
    // multimesh.add(mesh_66);
    // multimesh.add(mesh_67);
    // multimesh.add(mesh_68);
    // multimesh.add(mesh_69);
    // multimesh.add(mesh_70);
    // multimesh.add(mesh_71);
    // multimesh.add(mesh_72);
    // multimesh.add(mesh_73);
    // multimesh.add(mesh_74);
    // multimesh.add(mesh_75);
    // multimesh.add(mesh_76);
    // multimesh.add(mesh_77);
    // multimesh.add(mesh_78);
    // multimesh.add(mesh_79);
    // multimesh.add(mesh_80);
    // multimesh.add(mesh_81);
    // multimesh.add(mesh_82);
    // multimesh.add(mesh_83);
    // multimesh.add(mesh_84);
    // multimesh.add(mesh_85);
    // multimesh.add(mesh_86);
    // multimesh.add(mesh_87);
    // multimesh.add(mesh_88);
    // multimesh.add(mesh_89);
    // multimesh.add(mesh_90);
    // multimesh.add(mesh_91);
    // multimesh.add(mesh_92);
    // multimesh.add(mesh_93);
    // multimesh.add(mesh_94);
    // multimesh.add(mesh_95);
    // multimesh.add(mesh_96);
    // multimesh.add(mesh_97);
    // multimesh.add(mesh_98);
    // multimesh.add(mesh_99);

    multimesh.build();

    tools::dolfin_write_medit_triangles("multimesh",multimesh);

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
    // std::cout << "try rotate " << md << " with angle " << v << std::endl;

    double x0,y0,x1,y1,b,c,d,e,f;
    if (rotation_inside(md.x0, md.y0, md.cx, md.cy, v, x0,y0) and
	rotation_inside(md.x0, md.y1, md.cx, md.cy, v, c,d) and
	rotation_inside(md.x1, md.y0, md.cx, md.cy, v, e,f) and
	rotation_inside(md.x1, md.y1, md.cx, md.cy, v, x1,y1))
    {
      mdnew = MeshData(md.x0,md.y0, md.x1,md.y1, md.m, md.n, md.v+v);
      // std::cout << "rotated mesh found"<<std::endl;
      return true;
    }
    else return false;
  }

  void test_multiple_meshes_with_dynamic_rotation()
  {
    //set_log_level(DEBUG);
    dolfin::seed(0);

    const double h = 0.1;
    UnitSquareMesh background_mesh((int)std::round(1./h),
				   (int)std::round(1./h));

    // Create data for Nmeshes
    const std::size_t Nmeshes = 10;
    std::vector<MeshData> md(Nmeshes);
    std::size_t i = 0;
    std::cout << "Create initial meshes\n";
    while (i < Nmeshes)
    {
      const double x0 = dolfin::rand();
      const double x1 = dolfin::rand();
      const double y0 = dolfin::rand();
      const double y1 = dolfin::rand();
      const double v = dolfin::rand()*90; // initial rotation
      MeshData mdold(x0,y0, x1,y1,
		     std::max((int)std::round((x1-x0)/h), 1),
		     std::max((int)std::round((y1-y0)/h), 1),
		     v);
      bool mesh_ok = find_rotated_mesh(mdold, v, md[i]);
      if (mesh_ok)
      {
	std::cout << i << ' ' << md[i] << std::endl;
	i++;
      }
      else{std::cout << "try again\n"; }

      // rotate(x0, y0, cx, cy, v, xr, yr);
      // if (xr > 0 and xr < 1 and yr > 0 and yr < 1)
      // {
      // 	rotate(x0, y1, cx, cy, v, xr, yr);
      // 	if (xr > 0 and xr < 1 and yr > 0 and yr < 1)
      // 	{
      // 	  rotate(x1, y0, cx, cy, v, xr, yr);
      // 	  if (xr > 0 and xr < 1 and yr > 0 and yr < 1)
      // 	  {
      // 	    rotate(x1, y1, cx, cy, v, xr, yr);
      // 	    if (xr > 0 and xr < 1 and yr > 0 and yr < 1)
      // 	    {
      // 	      md[i] = MeshData(x0, x1, y0, y1,
      // 			       std::max((int)std::round((x1-x0)/h), 1),
      // 			       std::max((int)std::round((y1-y0)/h), 1),
      // 			       v);
      // 	      std::cout << i << ' ' << md[i] << std::endl;
      // 	      i++;
      // 	    }
      // 	  }
      // 	}
      // }
    }


    // {
    //   MultiMesh multimesh;
    //   multimesh.add(background_mesh);
    //   RectangleMesh mesh_0(md[0].x0, md[0].y0, md[0].x1, md[0].y1, md[0].m, md[0].n);
    //   mesh_0.rotate(md[0].v);
    //   multimesh.add(mesh_0);
    //   multimesh.build();
    //   tools::dolfin_write_medit_triangles("before_multimesh",multimesh, 0);
    //   PPause;
    // }

    // Create rotations
    const std::size_t Nangles = 180;
    const double angle_step = 180. / Nangles;

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
	  mesh_ok = find_rotated_mesh(md[i], -angle_step, mdnew);
	}

	if (!mesh_ok)
	{
	  std::cout << i<<' '<<md[i] <<std::endl;
	  PPause;
	} // should work

	if (mesh_ok)
	  md[i] = mdnew;
      }

      RectangleMesh mesh_0(md[0].x0, md[0].y0, md[0].x1, md[0].y1, md[0].m, md[0].n);
      mesh_0.rotate(md[0].v);
      RectangleMesh mesh_1(md[1].x0, md[1].y0, md[1].x1, md[1].y1, md[1].m, md[1].n);
      mesh_1.rotate(md[1].v);
      RectangleMesh mesh_2(md[2].x0, md[2].y0, md[2].x1, md[2].y1, md[2].m, md[2].n);
      mesh_2.rotate(md[2].v);
      RectangleMesh mesh_3(md[3].x0, md[3].y0, md[3].x1, md[3].y1, md[3].m, md[3].n);
      mesh_3.rotate(md[3].v);
      RectangleMesh mesh_4(md[4].x0, md[4].y0, md[4].x1, md[4].y1, md[4].m, md[4].n);
      mesh_4.rotate(md[4].v);
      RectangleMesh mesh_5(md[5].x0, md[5].y0, md[5].x1, md[5].y1, md[5].m, md[5].n);
      mesh_5.rotate(md[5].v);
      RectangleMesh mesh_6(md[6].x0, md[6].y0, md[6].x1, md[6].y1, md[6].m, md[6].n);
      mesh_6.rotate(md[6].v);
      RectangleMesh mesh_7(md[7].x0, md[7].y0, md[7].x1, md[7].y1, md[7].m, md[7].n);
      mesh_7.rotate(md[7].v);
      RectangleMesh mesh_8(md[8].x0, md[8].y0, md[8].x1, md[8].y1, md[8].m, md[8].n);
      mesh_8.rotate(md[8].v);
      RectangleMesh mesh_9(md[9].x0, md[9].y0, md[9].x1, md[9].y1, md[9].m, md[9].n);
      mesh_9.rotate(md[9].v);

      multimesh.add(mesh_0);
      multimesh.add(mesh_1);
      multimesh.add(mesh_2);
      multimesh.add(mesh_3);
      // multimesh.add(mesh_4);
      // multimesh.add(mesh_5);
      // multimesh.add(mesh_6);
      // multimesh.add(mesh_7);
      // multimesh.add(mesh_8);
      // multimesh.add(mesh_9);

      multimesh.build();

      tools::dolfin_write_medit_triangles("after_multimesh",multimesh, j);

      // Exact volume is known
      const double exact_volume = 1;
      const double volume = compute_volume(multimesh, exact_volume);

      //CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
    }

  }

  //------------------------------------------------------------------------------
  void test_exclusion_inclusion()
  {
    set_log_level(DEBUG);

    const double v = 1e-16;

    UnitSquareMesh mesh_0(1, 1);
    RectangleMesh mesh_1(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    mesh_1.rotate(v, 2);

    RectangleMesh mesh_2(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
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


//   void test_multiple_meshes_interface_quadrature()
//   {
//     // // These three meshes are ok
//     // UnitSquareMesh mesh_0(1, 1);
//     // RectangleMesh mesh_1(0.1, 0.1, 0.9, 0.9, 1, 1);
//     // RectangleMesh mesh_2(0.2, 0.2, 0.8, 0.8, 1, 1);
//     // double exact_volume = 4*(0.9-0.1); // mesh0 and mesh1
//     // exact_volume += 4*(0.8-0.2); // mesh1 and mesh2


//     // UnitCubeMesh mesh_0(1, 2, 3);
//     // BoxMesh mesh_1(0.1, 0.1, 0.1,    0.9, 0.9, 0.9,   2,3,4);//2, 3, 4);
//     // BoxMesh mesh_2(-0.1, -0.1, -0.1,    0.7, 0.7, 0.7,   4, 3, 2);
//     // BoxMesh mesh_3(0.51, 0.51, 0.51,    0.7, 0.7, 0.7,   1,1,1);//4, 3, 2);
//     // BoxMesh mesh_4(0.3, 0.3, 0.3,    0.7, 0.7, 0.7,   1,1,1);
//     // double exact_volume = 0.8*0.8*6; // for mesh_0 and mesh_1
//     // exact_volume += 0.4*0.4*6; // for mesh_1 and mesh_4


//     UnitCubeMesh mesh_0(1, 1, 1);
//     BoxMesh mesh_1(0.1, 0.1, 0.1,    0.9, 0.9, 0.9,   1, 1, 1);
//     BoxMesh mesh_2(0.2, 0.2, 0.2,    0.8, 0.8, 0.8,   1, 1, 1);
//     // BoxMesh mesh_3(0.51, 0.51, 0.51,    0.7, 0.7, 0.7,   1,1,1);//4, 3, 2);
//     // BoxMesh mesh_4(0.3, 0.3, 0.3,    0.7, 0.7, 0.7,   1,1,1);
//     double exact_volume = (0.9-0.1)*(0.9-0.1)*6; // for mesh_0 and mesh_1
//     exact_volume += (0.8-0.2)*(0.8-0.2)*6; // mesh_1 and mesh_2



//     // UnitCubeMesh mesh_0(1, 1, 1);
//     // MeshEditor editor;
//     // Mesh mesh_1;
//     // editor.open(mesh_1, 3, 3);
//     // editor.init_vertices(4);
//     // editor.init_cells(1);
//     // editor.add_vertex(0, Point(0.7, 0.1, -0.1));
//     // editor.add_vertex(1, Point(0.7, 0.3, -0.1));
//     // editor.add_vertex(2, Point(0.5, 0.1, -0.1));
//     // editor.add_vertex(3, Point(0.7, 0.1, 0.1));
//     // editor.add_cell(0, 0,1,2,3);
//     // editor.close();

//     // Mesh mesh_2;
//     // editor.open(mesh_2, 3,3);
//     // editor.init_vertices(4);
//     // editor.init_cells(1);
//     // editor.add_vertex(0, Point(0.7, 0.1, -0.2));
//     // editor.add_vertex(1, Point(0.7, 0.3, -0.2));
//     // editor.add_vertex(2, Point(0.5, 0.1, -0.2));
//     // editor.add_vertex(3, Point(0.7, 0.1, 0.05));
//     // editor.add_cell(0, 0,1,2,3);
//     // editor.close();

//     //double exact_volume = 0.8*0.8*6; // for mesh_0 and mesh_1
//     //exact_volume += 0.4*0.4*6; // for mesh_1 and mesh_4


//     // MeshEditor editor;
//     // Mesh mesh_0;
//     // editor.open(mesh_0, 2, 2);
//     // editor.init_vertices(3);
//     // editor.init_cells(1);
//     // editor.add_vertex(0, Point(0.,0.));
//     // editor.add_vertex(1, Point(2.,0.));
//     // editor.add_vertex(2, Point(1.,2.));
//     // editor.add_cell(0, 0,1,2);
//     // editor.close();

//     // Mesh mesh_1;
//     // editor.open(mesh_1, 2, 2);
//     // editor.init_vertices(3);
//     // editor.init_cells(1);
//     // editor.add_vertex(0, Point(0.,-0.5));
//     // editor.add_vertex(1, Point(2.,-0.5));
//     // editor.add_vertex(2, Point(1.,1.5));
//     // editor.add_cell(0, 0,1,2);
//     // editor.close();

//     // Mesh mesh_2;
//     // editor.open(mesh_2, 2, 2);
//     // editor.init_vertices(3);
//     // editor.init_cells(1);
//     // editor.add_vertex(0, Point(0.,-1.));
//     // editor.add_vertex(1, Point(2.,-1.));
//     // editor.add_vertex(2, Point(1.,1.));
//     // editor.add_cell(0, 0,1,2);
//     // editor.close();

//     // double exact_volume = 2*std::sqrt(0.75*0.75 + 1.5*1.5); // mesh_0 and mesh_1
//     // exact_volume += 2*std::sqrt(0.5*0.5 + 1*1); // mesh_0 and mesh_2
//     // exact_volume += 2*std::sqrt(0.75*0.75 + 1.5*1.5); // mesh_1and mesh_2
//     // double volume = 0;



//     // // These three meshes are ok.
//     // MeshEditor editor;
//     // Mesh mesh_0;
//     // editor.open(mesh_0, 2, 2);
//     // editor.init_vertices(3);
//     // editor.init_cells(1);
//     // editor.add_vertex(0, Point(0.,0.));
//     // editor.add_vertex(1, Point(2.,0.));
//     // editor.add_vertex(2, Point(1.,2.));
//     // editor.add_cell(0, 0,1,2);
//     // editor.close();

//     // Mesh mesh_1;
//     // editor.open(mesh_1, 2, 2);
//     // editor.init_vertices(3);
//     // editor.init_cells(1);
//     // editor.add_vertex(0, Point(1.5,-2.));
//     // editor.add_vertex(1, Point(4.,0.));
//     // editor.add_vertex(2, Point(1.5,2));
//     // editor.add_cell(0, 0,1,2);
//     // editor.close();

//     // Mesh mesh_2;
//     // editor.open(mesh_2, 2, 2);
//     // editor.init_vertices(3);
//     // editor.init_cells(1);
//     // editor.add_vertex(0, Point(3.,0.5));
//     // editor.add_vertex(1, Point(-1.,0.5));
//     // editor.add_vertex(2, Point(1.,-1.5));
//     // editor.add_cell(0, 0,1,2);
//     // editor.close();

//     // double exact_volume = (1.5-0.25) + (1-0.5); // mesh_0, mesh_1 and mesh_2
//     // exact_volume += (3-1.5) + std::sqrt(1.5*1.5 + 1.5*1.5); // mesh_1 and mesh_2


//     File("mesh_0.xml") << mesh_0;
//     File("mesh_1.xml") << mesh_1;
//     File("mesh_2.xml") << mesh_2;

//     // Build the multimesh
//     MultiMesh multimesh;
//     multimesh.add(mesh_0);
//     multimesh.add(mesh_1);
//     multimesh.add(mesh_2);
//     //multimesh.add(mesh_3);
//     //multimesh.add(mesh_4);
//     multimesh.build();


//     // Sum contribution from all parts
//     std::cout << "\n\n Sum up\n\n";
//     double volume = 0;
//     for (std::size_t part = 0; part < multimesh.num_parts(); part++)
//     {
//       std::cout << "% part " << part << '\n';
//       double part_volume = 0;

//       const auto& quadrature_rules = multimesh.quadrature_rule_interface(part);

//       // Get collision map
//       const auto& cmap = multimesh.collision_map_cut_cells(part);
//       for (auto it = cmap.begin(); it != cmap.end(); ++it)
//       {
//         const unsigned int cut_cell_index = it->first;

//         // Iterate over cutting cells
//         const auto& cutting_cells = it->second;
//         for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
//         {
//           //const std::size_t cutting_part = jt->first;
//           //const std::size_t cutting_cell_index = jt->second;

//           // Get quadrature rule for interface part defined by
//           // intersection of the cut and cutting cells
//           const std::size_t k = jt - cutting_cells.begin();
//           dolfin_assert(k < quadrature_rules.at(cut_cell_index).size());
//           const auto& qr = quadrature_rules.at(cut_cell_index)[k];

//           for (std::size_t j = 0; j < qr.second.size(); ++j)
//           {
//             volume += qr.second[j];
//             part_volume += qr.second[j];
//           }

//         }
//       }

//       std::cout<<"part volume " << part_volume<<std::endl;
//     }

//     std::cout << "exact volume " << exact_volume<<'\n'
//               << "volume " << volume<<std::endl;
//     CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
//   }




  //------------------------------------------------------------------------------
  void test_exclusion_inclusion_small_angle()
  {
    //set_log_level(DEBUG);

    exactinit();

    // UnitSquareMesh mesh_0(1, 1);
    // RectangleMesh mesh_1(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    // mesh_1.rotate(1e-14, 2);

    // MultiMesh multimesh;
    // multimesh.add(mesh_0);
    // multimesh.add(mesh_1);
    // multimesh.build();

    // tools::dolfin_write_medit_triangles("multimesh",multimesh);

    // const double exact_volume = 1;
    // const double volume = compute_volume(multimesh, exact_volume);

    {

      std::stringstream ss;
      ss << "angle_output_90.txt";
      std::ofstream file(ss.str());
      if (!file.good()) { std::cout << ss.str() << " not ok" << std::endl; exit(0); }
      file.precision(15);

      std::vector<double> angles;
      // double v = 100;
      // while (v > 1e-17)
      // {
      // 	angles.push_back(v);
      // 	v /= 10;
      // }
      angles.push_back(1e-7);
      // for (std::size_t i = 1; i < 90; ++i)
      // 	angles.push_back(i);

      double max_error = -1;

      for (const auto v: angles)
      {
	std::cout << "--------------------------------------\n"
		  << "try v = " << v << std::endl;
	for (std::size_t m = 3; m <= 3; ++m)
	  for (std::size_t n = 9; n <= 9; ++n)
	  {
	    UnitSquareMesh mesh_0(m, n);
	    RectangleMesh mesh_1(0.2, 0.2, 0.8, 0.8, m, n);
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
	//exit(0);
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

    // std::cout << "a=[";
    // for (const auto v: all_volumes)
    //   std::cout << std::setprecision(13)<< v <<' ';
    // std::cout << "]; plot(diff(a(2:end-1)),'x-');\n";


    // std::cout << std::setprecision(13)
    // 	      << "exact volume " << exact_volume << '\n'
    // 	      << "volume " << volume << '\n'
    // 	      << "error " << exact_volume - volume << '\n'
    // 	      << std::endl;

    return volume;
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

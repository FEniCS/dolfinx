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
#include <predicates.h>

using namespace dolfin;

class MultiMeshes : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MultiMeshes);
  //CPPUNIT_TEST(test_exclusion_inclusion);
  CPPUNIT_TEST(test_exclusion_inclusion_small_angle);
  //CPPUNIT_TEST(test_multiple_meshes_quadrature);
  //CPPUNIT_TEST(test_multiple_meshes_interface_quadrature);
  CPPUNIT_TEST_SUITE_END();

public:

  //------------------------------------------------------------------------------
  void test_exclusion_inclusion()
  {
    set_log_level(DEBUG);

    UnitSquareMesh mesh_0(1, 1);
    RectangleMesh mesh_1(0.200000, 0.200000, 0.800000, 0.800000, 1, 1);
    mesh_1.rotate(1e-8, 2);

    RectangleMesh mesh_2(0.300000, 0.300000, 0.700000, 0.700000, 1, 1);
    //mesh_2.rotate(8.002805e-01, 2);
    //mesh_2.rotate(1.418863e-01, 2);
    mesh_2.rotate(2e-8, 2);

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
    set_log_level(DEBUG);

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
      double v = 1e-12;//100;
      while (v > 1e-17)
      {
	std::cout << "--------------------------------------\n"
		  << "try v = " << v << std::endl;
	for (std::size_t m = 5; m <= 5; ++m)
	  for (std::size_t n = 5; n <= 5; ++n)
	  {
	    UnitSquareMesh mesh_0(m, n);
	    RectangleMesh mesh_1(0.200000, 0.200000, 0.800000, 0.800000, m, n);
	    mesh_1.rotate(v, 2);

	    MultiMesh multimesh;
	    multimesh.add(mesh_0);
	    multimesh.add(mesh_1);
	    multimesh.build();

	    tools::dolfin_write_medit_triangles("multimesh",multimesh);

	    const double exact_volume = 1;
	    const double volume = compute_volume(multimesh, exact_volume);
	    const double e = std::abs(volume - exact_volume);
	    std::cout << "v = " << v << '\n'
		      << "m = " << m << '\n'
		      << "n = " << n << '\n'
		      << "volume = " << volume << '\n'
		      << "error = " << e << '\n';

	    CPPUNIT_ASSERT_DOUBLES_EQUAL(exact_volume, volume, DOLFIN_EPS_LARGE);
	  }
	v /= 10;
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
	  file << qr.second[i]<<'\n';
          volume += qr.second[i];
          part_volume += qr.second[i];
        }
	status[*it] = 2;
      }
      std::cout << "\ttotal volume " << part_volume << std::endl;
      all_volumes.push_back(part_volume);

      tools::dolfin_write_medit_triangles("status",*multimesh.part(part),part);
      tools::dolfin_write_bb("status",*multimesh.part(part),status,part);

    }
    file.close();

    std::cout << "a=[";
    for (const auto v: all_volumes)
      std::cout << std::setprecision(13)<< v <<' ';
    std::cout << "]; plot(diff(a(2:end-1)),'x-');\n";


    std::cout << std::setprecision(13)
	      << "exact volume " << exact_volume << '\n'
	      << "volume " << volume << '\n'
	      << "error " << exact_volume - volume << '\n'
	      << std::endl;

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

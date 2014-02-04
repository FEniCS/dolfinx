// Copyright (C) 2014 Anders Logg
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
// First added:  2014-02-03
// Last changed: 2014-02-04

#include <dolfin/mesh/MeshEntity.h>
#include "Point.h"
#include "CollisionDetection.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
bool
CollisionDetection::collides_edge_edge(const Point& a,
				       const Point& b,
				       const Point& c,
				       const Point& d)
{
  // Form edges and normal
  const Point L1=b-a, L2=d-c;
  const Point n=L1.cross(L2);

  const Point ca=c-a;

  // L1 and L2 must be coplanar for collision
  if (std::abs(ca.dot(n))>DOLFIN_EPS) return false;

  // Find orthogonal plane with normal nplane
  const Point nplane=n.cross(L1);

  // Check if c and d are on opposite sides of this plane
  if (nplane.dot(ca)*nplane.dot(d-a)>DOLFIN_EPS) return false;

  return true;


  // // something like nplane.dot(c-a)*nplane.dot(d-a).
  // //It is however more stable to do

  // // const double nplanea=nplane.dot(a);
  // // if ((nplane.dot(c)-nplanea) * (nplane.dot(d)-nplanea) > ) return false;







  // // Algorithm from Real-time collision detection by Christer Ericson:
  // // Test2DSegmentSegment on page 152, Section 5.1.9.

  // // Compute signed areas of abd and abc
  // const double abd = signed_area(a, b, d);
  // const double abc = signed_area(a, b, c);

  // // Return false if not intersecting (or collinear)
  // if (abd*abc >= 0.0)
  //   return false;

  // // Compute signed area of cda
  // const double cda = signed_area(c, d, a);

  // // Check whether segments collide
  // return cda*(cda + abc - abd) < -DOLFIN_EPS;

}
//-----------------------------------------------------------------------------
bool
CollisionDetection::collides_triangle_point(const MeshEntity& triangle,
					    const Point& point)
{
  // Algorithm from http://www.blackpawn.com/texts/pointinpoly/
  // See also "Real-Time Collision Detection" by Christer Ericson.
  //
  // We express AP as a linear combination of the vectors AB and
  // AC. Point is inside triangle iff AP is a convex combination.
  //
  // Note: This function may be optimized if only 2D vectors and inner
  // products need to be computed.

  dolfin_assert(triangle.mesh().topology().dim() == 2);

  // Get the vertices as points
  const MeshGeometry& geometry = triangle.mesh().geometry();
  const unsigned int* vertices = triangle.entities(0);
  const Point p0 = geometry.point(vertices[0]);
  const Point p1 = geometry.point(vertices[1]);
  const Point p2 = geometry.point(vertices[2]);

  // Compute vectors
  const Point v1 = p1 - p0;
  const Point v2 = p2 - p0;
  const Point v = point - p0;

  // Compute entries of linear system
  const double a11 = v1.dot(v1);
  const double a12 = v1.dot(v2);
  const double a22 = v2.dot(v2);
  const double b1 = v.dot(v1);
  const double b2 = v.dot(v2);

  // Solve linear system
  const double inv_det = 1.0 / (a11*a22 - a12*a12);
  const double x1 = inv_det*( a22*b1 - a12*b2);
  const double x2 = inv_det*(-a12*b1 + a11*b2);

  // Tolerance for numeric test (using vector v1)
  const double dx = std::abs(v1.x());
  const double dy = std::abs(v1.y());
  const double eps = std::max(DOLFIN_EPS_LARGE, DOLFIN_EPS_LARGE*std::max(dx, dy));

  // Check if point is inside
  return x1 >= -eps && x2 >= -eps && x1 + x2 <= 1.0 + eps;
}
//-----------------------------------------------------------------------------
bool
CollisionDetection::collides_triangle_triangle(const MeshEntity& triangle_0,
					       const MeshEntity& triangle_1)
{
  // This algorithm and code is from
  // Triangle/triangle intersection test routine,
  // by Tomas Moller, 1997.
  // See article "A Fast Triangle-Triangle Intersection Test",
  // Journal of Graphics Tools, 2(2), 1997

  dolfin_assert(triangle_0.mesh().topology().dim() == 2);
  dolfin_assert(triangle_1.mesh().topology().dim() == 2);

  // Get vertices as points
  const MeshGeometry& geometry_0 = triangle_0.mesh().geometry();
  const unsigned int* vertices_0 = triangle_0.entities(0);
  const Point V0=geometry_0.point(vertices_0[0]);
  const Point V1=geometry_0.point(vertices_0[1]);
  const Point V2=geometry_0.point(vertices_0[2]);

  const MeshGeometry& geometry_1 = triangle_1.mesh().geometry();
  const unsigned int* vertices_1 = triangle_1.entities(0);
  const Point U0=geometry_1.point(vertices_1[0]);
  const Point U1=geometry_1.point(vertices_1[1]);
  const Point U2=geometry_1.point(vertices_1[2]);


  // double E1[3],E2[3];
  // double N1[3],N2[3],d1,d2;
  // double du0,du1,du2,dv0,dv1,dv2;
  // double D[3];
  // double isect1[2], isect2[2];
  // double du0du1,du0du2,dv0dv1,dv0dv2;
  // short index;
  // double vp0,vp1,vp2;
  // double up0,up1,up2;
  // double bb,cc,max;

  // compute plane equation of triangle(V0,V1,V2)
  // SUB(E1,V1,V0);
  // SUB(E2,V2,V0);
  // CROSS(N1,E1,E2);
  // d1=-DOT(N1,V0);
  Point E1=V1-V0;
  Point E2=V2-V0;
  const Point N1=E1.cross(E2);
  const double d1=-N1.dot(V0);

  // plane equation 1: N1.X+d1=0 //

  // put U0,U1,U2 into plane equation 1 to compute signed distances to the plane
  // du0=DOT(N1,U0)+d1;
  // du1=DOT(N1,U1)+d1;
  // du2=DOT(N1,U2)+d1;
  // double du0=N1.dot(U0)+d1;
  // double du1=N1.dot(U1)+d1;
  // double du2=N1.dot(U2)+d1;
  Point du(N1.dot(U0)+d1, N1.dot(U1)+d1, N1.dot(U2)+d1);

//   // coplanarity robustness check
// #if USE_EPSILON_TEST==TRUE
//   if(FABS(du0)<EPSILON) du0=0.0;
//   if(FABS(du1)<EPSILON) du1=0.0;
//   if(FABS(du2)<EPSILON) du2=0.0;
// #endif
//   du0du1=du0*du1;
//   du0du2=du0*du2;
  if (std::abs(du[0])<DOLFIN_EPS) du[0]=0.;
  if (std::abs(du[1])<DOLFIN_EPS) du[1]=0.;
  if (std::abs(du[2])<DOLFIN_EPS) du[2]=0.;
  const double du0du1=du[0]*du[1];
  const double du0du2=du[0]*du[2];

  if (du0du1>0. && du0du2>0.) // same sign on all of them + not equal 0 ?
    return false;             // no intersection occurs

  // compute plane of triangle (U0,U1,U2)
  // SUB(E1,U1,U0);
  // SUB(E2,U2,U0);
  // CROSS(N2,E1,E2);
  // d2=-DOT(N2,U0);
  E1=U1-U0;
  E2=U2-U0;
  const Point N2=E1.cross(E2);
  const double d2=-N2.dot(U0);

  // plane equation 2: N2.X+d2=0
  // put V0,V1,V2 into plane equation 2
  // dv0=DOT(N2,V0)+d2;
  // dv1=DOT(N2,V1)+d2;
  // dv2=DOT(N2,V2)+d2;
  // double dv0=N2.dot(V0)+d2;
  // double dv1=N2.dot(V1)+d2;
  // double dv2=N2.dot(V2)+d2;
  Point dv(N2.dot(V0)+d2, N2.dot(V1)+d2, N2.dot(V2)+d2);

// #if USE_EPSILON_TEST==TRUE
//   if(FABS(dv0)<EPSILON) dv0=0.0;
//   if(FABS(dv1)<EPSILON) dv1=0.0;
//   if(FABS(dv2)<EPSILON) dv2=0.0;
// #endif
  if (std::abs(dv[0])<DOLFIN_EPS) dv[0]=0.;
  if (std::abs(dv[1])<DOLFIN_EPS) dv[1]=0.;
  if (std::abs(dv[2])<DOLFIN_EPS) dv[2]=0.;
  const double dv0dv1=dv[0]*dv[1];
  const double dv0dv2=dv[0]*dv[2];

  if (dv0dv1>0. && dv0dv2>0.) // same sign on all of them + not equal 0 ?
    return false;             // no intersection occurs

  // compute direction of intersection line
  //CROSS(D,N1,N2);
  const Point D=N1.cross(N2);

  // compute and index to the largest component of D
  double max=std::abs(D[0]);
  int index=0;
  const double bb=std::abs(D[1]);
  const double cc=std::abs(D[2]);
  if (bb>max) max=bb,index=1;
  if (cc>max) max=cc,index=2;

  // this is the simplified projection onto L
  const Point vp(V0[index],V1[index],V2[index]);
  const Point up(U0[index],U1[index],U2[index]);
  // const double vp0=V0[index];
  // const double vp1=V1[index];
  // const double vp2=V2[index];
  // const double up0=U0[index];
  // const double up1=U1[index];
  // const double up2=U2[index];

  // compute interval for triangle 1
  //double a,b,c,x0,x1;
  //NEWCOMPUTE_INTERVALS(vp0,vp1,vp2,dv0,dv1,dv2,dv0dv1,dv0dv2,a,b,c,x0,x1);
  Point abc;
  double x0,x1;
  if (compute_intervals(N1,V0,V1,V2,U0,U1,U2,
			vp,dv,dv0dv1,dv0dv2,abc,x0,x1)) return false;

  // compute interval for triangle 2
  //double d,e,f,y0,y1;
  //NEWCOMPUTE_INTERVALS(up0,up1,up2,du0,du1,du2,du0du1,du0du2,d,e,f,y0,y1);
  Point def;
  double y0,y1;
  if (compute_intervals(N1,V0,V1,V2,U0,U1,U2,
			up,du,du0du1,du0du2,def,y0,y1)) return false;

  const double xx=x0*x1;
  const double yy=y0*y1;
  const double xxyy=xx*yy;
  const double tmp1=abc[0]*xxyy;
  double isect1[2]={tmp1+abc[1]*x1*yy,
		    tmp1+abc[2]*x0*yy};

  const double tmp2=def[0]*xxyy;
  double isect2[2]={tmp2+def[1]*xx*y1,
		    tmp2+def[2]*xx*y0};

  // SORT(isect1[0],isect1[1]);
  // SORT(isect2[0],isect2[1]);
  if (isect1[0]>isect1[1]) std::swap(isect1[0],isect1[1]);
  if (isect2[0]>isect2[1]) std::swap(isect2[0],isect2[1]);

  if (isect1[1]<isect2[0] || isect2[1]<isect1[0]) return false;

  return true;





  // Do we need to check geometry dimension?

  // This is only implemented for triangle-triangle collisions at this point
  // if (entity.dim() != 2)
  // {
  //   dolfin_error("TriangleCell.cpp",
  //                "compute collision with entity",
  //                "Only know how to compute triangle-triangle collisions");
  // }

  // Get the vertices as points
  const MeshGeometry& geometry_p = triangle_0.mesh().geometry();
  const unsigned int* vertices_p = triangle_0.entities(0);
  const Point p0 = geometry_p.point(vertices_p[0]);
  const Point p1 = geometry_p.point(vertices_p[1]);
  const Point p2 = geometry_p.point(vertices_p[2]);

  // Get the vertices as points
  const MeshGeometry& geometry_q = triangle_1.mesh().geometry();
  const unsigned int* vertices_q = triangle_1.entities(0);
  const Point q0 = geometry_q.point(vertices_q[0]);
  const Point q1 = geometry_q.point(vertices_q[1]);
  const Point q2 = geometry_q.point(vertices_q[2]);


  // First check if triangles are completely overlapping (necessary
  // since tests below will fail for collinear edges). Note that this
  // test will also cover a few other cases with coinciding midpoints.
  const double eps2 = DOLFIN_EPS_LARGE*DOLFIN_EPS_LARGE*p0.squared_distance(p1);
  if (triangle_0.midpoint().squared_distance(triangle_1.midpoint()) < eps2)
    return true;

  // Check for pairwise collisions between the edges
  if (collides_edge_edge(p0, p1, q0, q1)) return true;
  if (collides_edge_edge(p0, p1, q1, q2)) return true;
  if (collides_edge_edge(p0, p1, q2, q0)) return true;
  if (collides_edge_edge(p1, p2, q0, q1)) return true;
  if (collides_edge_edge(p1, p2, q1, q2)) return true;
  if (collides_edge_edge(p1, p2, q2, q0)) return true;
  if (collides_edge_edge(p2, p0, q0, q1)) return true;
  if (collides_edge_edge(p2, p0, q1, q2)) return true;
  //if (collides(p2, p0, q2, q0)) return true; // optimization, not needed

  return false;
}
//-----------------------------------------------------------------------------
bool
CollisionDetection::collides_tetrahedron_point(const MeshEntity& tetrahedron,
                                               const Point& point)
{
  dolfin_assert(tetrahedron.mesh().topology().dim() == 3);

  dolfin_not_implemented();
  return false;
}
//-----------------------------------------------------------------------------
bool
CollisionDetection::collides_tetrahedron_triangle(const MeshEntity& tetrahedron,
                                                  const MeshEntity& triangle)
{
  dolfin_assert(tetrahedron.mesh().topology().dim() == 3);
  dolfin_assert(triangle.mesh().topology().dim() == 2);

  dolfin_not_implemented();
  return false;
}
//-----------------------------------------------------------------------------
bool
CollisionDetection::collides_tetrahedron_tetrahedron(const MeshEntity& tetrahedron_0,
                                                     const MeshEntity& tetrahedron_1)
{
  // This algorithm checks whether two tetrahedra intersects.

  // Algorithm and source code from Fabio Ganovelli, Federico Ponchio
  // and Claudio Rocchini: Fast Tetrahedron-Tetrahedron Overlap
  // Algorithm. DOI: 10.1080/10867651.2002.10487557.

  dolfin_assert(tetrahedron_0.mesh().topology().dim() == 3);
  dolfin_assert(tetrahedron_1.mesh().topology().dim() == 3);

  // Get the vertices as points
  const MeshGeometry& geometry = tetrahedron_0.mesh().geometry();
  const unsigned int* vertices = tetrahedron_0.entities(0);
  const MeshGeometry& geometry_q = tetrahedron_1.mesh().geometry();
  const unsigned int* vertices_q = tetrahedron_1.entities(0);
  std::vector<Point> V1(4),V2(4);
  for (int i=0; i<4; ++i)
  {
    V1[i]=geometry.point(vertices[i]);
    V2[i]=geometry_q.point(vertices_q[i]);
  }

  // Get the vectors between V2 and V1[0]
  std::vector<Point> P_V1(4);
  for (int i=0; i<4; ++i)
    P_V1[i] = V2[i]-V1[0];

  // Data structure for edges of V1 and V2
  std::vector<Point> e_v1(5), e_v2(5);
  e_v1[0]=V1[1]-V1[0];
  e_v1[1]=V1[2]-V1[0];
  e_v1[2]=V1[3]-V1[0];

  Point n=e_v1[1].cross(e_v1[0]);

  // Maybe flip normal. Normal should be outward.
  if (n.dot(e_v1[2])>0) n*=-1;

  std::vector<int> masks(4);
  std::vector<std::vector<double> > Coord_1(4,std::vector<double>(4));

  if (separating_plane_face_A_1(P_V1,n, Coord_1[0],masks[0])) return false;

  n=e_v1[0].cross(e_v1[2]);
  // Maybe flip normal
  if (n.dot(e_v1[1])>0) n*=-1;

  if (separating_plane_face_A_1(P_V1,n, Coord_1[1],masks[1])) return false;

  if (separating_plane_edge_A(Coord_1,masks, 0,1)) return false;

  n=e_v1[2].cross(e_v1[1]);

  // Mmaybe flip normal
  if (n.dot(e_v1[0])>0) n*=-1;

  if (separating_plane_face_A_1(P_V1,n, Coord_1[2],masks[2])) return false;

  if (separating_plane_edge_A(Coord_1,masks, 0,2)) return false;

  if (separating_plane_edge_A(Coord_1,masks, 1,2)) return false;

  e_v1[4]=V1[3]-V1[1];
  e_v1[3]=V1[2]-V1[1];

  n=e_v1[3].cross(e_v1[4]);

  // Maybe flip normal. Note the < since e_v1[0]=v1-v0.
  if (n.dot(e_v1[0])<0) n*=-1;

  if (separating_plane_face_A_2(V1,V2,n, Coord_1[3],masks[3])) return false;

  if (separating_plane_edge_A(Coord_1,masks, 0,3)) return false;

  if (separating_plane_edge_A(Coord_1,masks, 1,3)) return false;

  if (separating_plane_edge_A(Coord_1,masks, 2,3)) return false;

  if ((masks[0] | masks[1] | masks[2] | masks[3] )!= 15) return true;

  // From now on, if there is a separating plane, it is parallel to a
  // face of b.

  std::vector<Point> P_V2(4);
  for (int i=0; i<4; ++i)
    P_V2[i] = V1[i]-V2[0];

  e_v2[0]=V2[1]-V2[0];
  e_v2[1]=V2[2]-V2[0];
  e_v2[2]=V2[3]-V2[0];

  n=e_v2[1].cross(e_v2[0]);

  // Maybe flip normal
  if (n.dot(e_v2[2])>0) n*=-1;

  if (separating_plane_face_B_1(P_V2,n)) return false;

  n=e_v2[0].cross(e_v2[2]);

  // Maybe flip normal
  if (n.dot(e_v2[1])>0) n*=-1;

  if (separating_plane_face_B_1(P_V2,n)) return false;

  n=e_v2[2].cross(e_v2[1]);

  // Maybe flip normal
  if (n.dot(e_v2[0])>0) n*=-1;

  if (separating_plane_face_B_1(P_V2,n)) return false;

  e_v2[4]=V2[3]-V2[1];
  e_v2[3]=V2[2]-V2[1];

  n=e_v2[3].cross(e_v2[4]);

  // Maybe flip normal. Note the < since e_v2[0]=V2[1]-V2[0].
  if (n.dot(e_v2[0])<0) n*=-1;

  if (separating_plane_face_B_2(V1,V2,n)) return false;

  return true;
}
//-----------------------------------------------------------------------------
bool
CollisionDetection::compute_intervals(const Point& N1,
				      const Point& V0,
				      const Point& V1,
				      const Point& V2,
				      const Point& U0,
				      const Point& U1,
				      const Point& U2,
				      const Point& VV,
				      const Point& D,
				      double D0D1,
				      double D0D2,
				      Point& ABC,
				      double& X0,
				      double& X1)
{
  if(D0D1>0.)
  {
    // here we know that D0D2<=0.0
    // that is D[0], D[1] are on the same side, D2 on the other or on the plane
    ABC[0]=VV[2];
    ABC[1]=(VV[0]-VV[2])*D[2];
    ABC[2]=(VV[1]-VV[2])*D[2];
    X0=D[2]-D[0];
    X1=D[2]-D[1];
  }
  else if (D0D2>0.)
  {
    // here we know that d0d1<=0.0
    ABC[0]=VV[1];
    ABC[1]=(VV[0]-VV[1])*D[1];
    ABC[2]=(VV[2]-VV[1])*D[1];
    X0=D[1]-D[0];
    X1=D[1]-D[2];
  }
  else if (D[1]*D[2]>0. || D[0]!=0.)
  {
    // here we know that d0d1<=0.0 or that D[0]!=0.0
    ABC[0]=VV[0];
    ABC[1]=(VV[1]-VV[0])*D[0];
    ABC[2]=(VV[2]-VV[0])*D[0];
    X0=D[0]-D[1];
    X1=D[0]-D[2];
  }
  else if (D[1]!=0.)
  {
    ABC[0]=VV[1];
    ABC[1]=(VV[0]-VV[1])*D[1];
    ABC[2]=(VV[2]-VV[1])*D[1];
    X0=D[1]-D[0];
    X1=D[1]-D[2];
  }
  else if (D[2]!=0.)
  {
    ABC[0]=VV[2];
    ABC[1]=(VV[0]-VV[2])*D[2];
    ABC[2]=(VV[1]-VV[2])*D[2];
    X0=D[2]-D[0];
    X1=D[2]-D[1];
  }
  else
  {
    // triangles are coplanar
    return coplanar_triangle_triangle(N1,V0,V1,V2,U0,U1,U2);
  }
  return false;
}
//-----------------------------------------------------------------------------
bool
CollisionDetection::coplanar_triangle_triangle(const Point& N,
					       const Point& V0,
					       const Point& V1,
					       const Point& V2,
					       const Point& U0,
					       const Point& U1,
					       const Point& U2)
{
  // First project onto an axis-aligned plane that maximizes the area
  // of the triangles. Then compute indices: i0,i1.
  const Point A(std::abs(N[0]),std::abs(N[1]),std::abs(N[2]));
  int i0,i1;

  if(A[0]>A[1])
  {
    if(A[0]>A[2])
    {
      i0=1;      // A[0] is greatest
      i1=2;
    }
    else
    {
      i0=0;      // A[2] is greatest
      i1=1;
    }
  }
  else   // A[0]<=A[1]
  {
    if(A[2]>A[1])
    {
      i0=0;      // A[2] is greatest
      i1=1;
    }
    else
    {
      i0=0;      // A[1] is greatest
      i1=2;
    }
  }

  // test all edges of triangle 1 against the edges of triangle 2
  if (edge_against_tri_edges(i0,i1,V0,V1,U0,U1,U2)) return true;
  if (edge_against_tri_edges(i0,i1,V1,V2,U0,U1,U2)) return true;
  if (edge_against_tri_edges(i0,i1,V2,V0,U0,U1,U2)) return true;

  // finally, test if tri1 is totally contained in tri2 or vice versa
  if (point_in_tri(i0,i1, V0, U0,U1,U2)) return true;
  if (point_in_tri(i0,i1, U0, V0,V1,V2)) return true;
  // if (collides_triangle_point(TriangleCell(U0,U1,U2),V0)) return true;
  // if (collides_triangle_point(TriangleCell(V0,V1,V2),U0)) return true;

  return false;
}
//-----------------------------------------------------------------------------
bool
CollisionDetection::edge_against_tri_edges(int i0,
					   int i1,
					   const Point& V0,
					   const Point& V1,
					   const Point& U0,
					   const Point& U1,
					   const Point& U2)
{
  const double Ax=V1[i0]-V0[i0];
  const double Ay=V1[i1]-V0[i1];

  // test edge U0,U1 against V0,V1 
  if (edge_edge_test(i0,i1,Ax,Ay, V0,U0,U1)) return true;
  
  // test edge U1,U2 against V0,V1 
  if (edge_edge_test(i0,i1,Ax,Ay, V0,U1,U2)) return true;
  
  // test edge U2,U1 against V0,V1 
  if (edge_edge_test(i0,i1,Ax,Ay, V0,U2,U0)) return true;
  
  return false;
}
//-----------------------------------------------------------------------------
bool
CollisionDetection::edge_edge_test(int i0,
				   int i1,
				   double Ax,
				   double Ay,
				   const Point& V0,
				   const Point& U0,
				   const Point& U1)
{
  // This edge to edge test is based on Franlin Antonio's gem: "Faster
  // Line Segment Intersection", in Graphics Gems III, pp. 199-202
  const double Bx=U0[i0]-U1[i0];
  const double By=U0[i1]-U1[i1];
  const double Cx=V0[i0]-U0[i0];
  const double Cy=V0[i1]-U0[i1];
  const double f=Ay*Bx-Ax*By;
  const double d=By*Cx-Bx*Cy;

  if ((f>0 && d>=0 && d<=f) || (f<0 && d<=0 && d>=f))
  {
    double e=Ax*Cy-Ay*Cx;
    if (f>0)
    {
      if (e>=0 && e<=f) return true;
    }
    else
    {
      if (e<=0 && e>=f) return true;
    }
  }
  return false;
}
//-----------------------------------------------------------------------------
bool 
CollisionDetection::point_in_tri(int i0,
				 int i1,
				 const Point& V0,
				 const Point& U0,
				 const Point& U1,
				 const Point& U2)
{
  double a=U1[i1]-U0[i1];
  double b=-(U1[i0]-U0[i0]);
  double c=-a*U0[i0]-b*U0[i1];
  double d0=a*V0[i0]+b*V0[i1]+c;

  a=U2[i1]-U1[i1];
  b=-(U2[i0]-U1[i0]);
  c=-a*U1[i0]-b*U1[i1];
  double d1=a*V0[i0]+b*V0[i1]+c;

  a=U0[i1]-U2[i1];
  b=-(U0[i0]-U2[i0]);
  c=-a*U2[i0]-b*U2[i1];
  double d2=a*V0[i0]+b*V0[i1]+c;
  if(d0*d1>0.0)
  {
    if(d0*d2>0.0) return true;
  }
  return false;
}
//-----------------------------------------------------------------------------
bool
CollisionDetection::separating_plane_face_A_1(const std::vector<Point>& P_V1,
					      const Point& n,
					      std::vector<double>& Coord,
					      int&  maskEdges)
{
  // Helper function for collides_tetrahedron: checks if plane pv1 is
  // a separating plane. Stores local coordinates bc and the mask bit
  // maskEdges.

  maskEdges = 0;
  if ((Coord[0] = P_V1[0].dot(n)) > 0) maskEdges = 1;
  if ((Coord[1] = P_V1[1].dot(n)) > 0) maskEdges |= 2;
  if ((Coord[2] = P_V1[2].dot(n)) > 0) maskEdges |= 4;
  if ((Coord[3] = P_V1[3].dot(n)) > 0) maskEdges |= 8;
  return (maskEdges == 15);
}

//-----------------------------------------------------------------------------
bool
CollisionDetection::separating_plane_face_A_2(const std::vector<Point>& V1,
					      const std::vector<Point>& V2,
					      const Point& n,
					      std::vector<double>& Coord,
					      int&  maskEdges)
{
  // Helper function for collides_tetrahedron: checks if plane v1,v2
  // is a separating plane. Stores local coordinates bc and the mask
  // bit maskEdges.

  maskEdges = 0;
  if ((Coord[0] = (V2[0]-V1[1]).dot(n)) > 0) maskEdges = 1;
  if ((Coord[1] = (V2[1]-V1[1]).dot(n)) > 0) maskEdges |= 2;
  if ((Coord[2] = (V2[2]-V1[1]).dot(n)) > 0) maskEdges |= 4;
  if ((Coord[3] = (V2[3]-V1[1]).dot(n)) > 0) maskEdges |= 8;
  return (maskEdges == 15);
}
//-----------------------------------------------------------------------------
bool
CollisionDetection::separating_plane_edge_A(const std::vector<std::vector<double> >& Coord_1,
					    const std::vector<int>& masks,
					    int f0,
					    int f1)
{
  // Helper function for collides_tetrahedron: checks if edge is in
  // the plane separating faces f0 and f1.

  const std::vector<double>& coord_f0=Coord_1[f0];
  const std::vector<double>& coord_f1=Coord_1[f1];

  int maskf0 = masks[f0];
  int maskf1 = masks[f1];

  if ((maskf0 | maskf1) != 15) // if there is a vertex of b
    return false; // included in (-,-) return false

  maskf0 &= (maskf0 ^ maskf1); // exclude the vertices in (+,+)
  maskf1 &= (maskf0 ^ maskf1);

  // edge 0: 0--1
  if (((maskf0 & 1) && // the vertex 0 of b is in (-,+)
       (maskf1 & 2)) && // the vertex 1 of b is in (+,-)
      (((coord_f0[1] * coord_f1[0]) -
        (coord_f0[0] * coord_f1[1])) > 0))
    // the edge of b (0,1) intersect (-,-) (see the paper)
    return false;

  if (((maskf0 & 2) && (maskf1 & 1)) && (((coord_f0[1] * coord_f1[0]) - (coord_f0[0] * coord_f1[1])) < 0))
    return false;

  // edge 1: 0--2
  if (((maskf0 & 1) && (maskf1 & 4)) && (((coord_f0[2] * coord_f1[0]) - (coord_f0[0] * coord_f1[2])) > 0))
    return false;

  if (((maskf0 & 4) && (maskf1 & 1)) && (((coord_f0[2] * coord_f1[0]) - (coord_f0[0] * coord_f1[2])) < 0))
    return false;

  // edge 2: 0--3
  if (((maskf0 & 1) &&(maskf1 & 8)) && (((coord_f0[3] * coord_f1[0]) - (coord_f0[0] * coord_f1[3])) > 0))
    return false;

  if (((maskf0 & 8) && (maskf1 & 1)) && (((coord_f0[3] * coord_f1[0]) - (coord_f0[0] * coord_f1[3])) < 0))
    return false;

  // edge 3: 1--2
  if (((maskf0 & 2) && (maskf1 & 4)) && (((coord_f0[2] * coord_f1[1]) - (coord_f0[1] * coord_f1[2])) > 0))
    return false;

  if (((maskf0 & 4) && (maskf1 & 2)) && (((coord_f0[2] * coord_f1[1]) - (coord_f0[1] * coord_f1[2])) < 0))
    return false;

  // edge 4: 1--3
  if (((maskf0 & 2) && (maskf1 & 8)) && (((coord_f0[3] * coord_f1[1]) - (coord_f0[1] * coord_f1[3])) > 0))
    return false;

  if (((maskf0 & 8) && (maskf1 & 2)) && (((coord_f0[3] * coord_f1[1]) - (coord_f0[1] * coord_f1[3])) < 0))
    return false;

  // edge 5: 2--3
  if (((maskf0 & 4) && (maskf1 & 8)) && (((coord_f0[3] * coord_f1[2]) - (coord_f0[2] * coord_f1[3])) > 0))
    return false;

  if (((maskf0 & 8) && (maskf1 & 4)) && (((coord_f0[3] * coord_f1[2]) - (coord_f0[2] * coord_f1[3])) < 0))
    return false;

  // Now there exists a separating plane supported by the edge shared
  // by f0 and f1.
  return true;
}
//-----------------------------------------------------------------------------

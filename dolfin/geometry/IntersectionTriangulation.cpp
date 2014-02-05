// Copyright (C) 2013 Anders Logg
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
// Last changed: 2014-02-03

#include <dolfin/mesh/MeshEntity.h>
#include "IntersectionTriangulation.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::vector<double>
triangulate_intersection_interval_interval(const MeshEntity& interval_0,
                                           const MeshEntity& interval_1)
{
  dolfin_assert(interval_0.mesh().topology().dim() == 1);
  dolfin_assert(interval_1.mesh().topology().dim() == 1);

  dolfin_not_implemented();
  std::vector<double> triangulation;
  return triangulation;
}
//-----------------------------------------------------------------------------
std::vector<double>
triangulate_intersection_triangulate_triangulate(const MeshEntity& triangle_0,
                                                 const MeshEntity& triangle_1)
{
  dolfin_assert(triangle_0.mesh().topology().dim() == 2);
  dolfin_assert(triangle_1.mesh().topology().dim() == 2);

  dolfin_not_implemented();
  std::vector<double> triangulation;
  return triangulation;
}
//-----------------------------------------------------------------------------
std::vector<double>
triangulate_intersection_tetrahedron_tetrahedron(const MeshEntity& tetrahedron_0,
                                                 const MeshEntity& tetrahedron_1)
{
  // This algorithm computes the intersection of cell_0 and cell_1 by
  // returning a vector<double> with points describing a tetrahedral
  // mesh of the intersection. We will use the fact that the
  // intersection is a convex polyhedron. The algorithm works by first
  // identifying intersection points: vertex points inside a cell,
  // edge-face collision points and edge-edge collision points (the
  // edge-edge is a rare occurance). Having the intersection points,
  // we identify points that are coplanar and thus form a facet of the
  // polyhedron. These points are then used to form a tesselation of
  // triangles, which are used to form tetrahedra by the use of the
  // center point of the polyhedron. This center point is thus an
  // additional point not found on the polyhedron facets.

  dolfin_assert(tetrahedron_0.mesh().topology().dim() == 3);
  dolfin_assert(tetrahedron_1.mesh().topology().dim() == 3);

  std::vector<double> triangulation;

  // // Tolerance for coplanar points
  // const double coplanar_tol=1e-11;

  // // Tolerance for the tetrahedron determinant (otherwise problems
  // // with warped tets)
  // const double tet_det_tol=1e-15; 

  // // Tolerance for duplicate points (p and q are the same if
  // // (p-q).norm() < same_point_tol)
  // const double same_point_tol=1e-15;

  // // Tolerance for small triangle (could be improved by identifying
  // // sliver and small triangles)
  // const double tri_det_tol=1e-11; 

  // // Points in the triangulation (unique)
  // std::vector<Point> points;

  // // Get the vertices as points
  // const MeshGeometry& geom_0 = tetrahedron_0.mesh().geometry();
  // const unsigned int* vert_0 = tetrahedron_0.entities(0);
  // const MeshGeometry& geom_1 = tetrahedron_1.mesh().geometry();
  // const unsigned int* vert_1 = tetrahedron_1.entities(0);

  // // Node intersection
  // for (int i=0; i<4; ++i) 
  // {
  //   if (collides_tetrahedron_point(tetrahedron_0, geom_1.point(vert_1[i]))) 
  //     points.push_back(geom_1.point(vert_1[i]));

  //   if (collides_tetrahedron_point(tetrahedron_1, geom_0.point(vert_0[i]))) 
  //     points.push_back(geom_0.point(vert_0[i]));
  // }

  // // Edge face intersections 
  // std::vector<std::vector<std::size_t> > edges_0(6,std::vector<std::size_t>(2));
  // std::vector<std::vector<std::size_t> > edges_1(6,std::vector<std::size_t>(2));
  // tetrahedron_0.create_entities(edges_0, 1,vert_0);
  // tetrahedron_1.create_entities(edges_1, 1,vert_1);

  // std::vector<std::vector<std::size_t> > faces_0(4,std::vector<std::size_t>(3));
  // std::vector<std::vector<std::size_t> > faces_1(4,std::vector<std::size_t>(3));
  // tetrahedron_0.create_entities(faces_0, 2,vert_0);
  // tetrahedron_1.create_entities(faces_1, 2,vert_1);

  // // Loop over edges e and faces f
  // for (int e=0; e<6; ++e) 
  // { 
  //   for (int f=0; f<4; ++f) 
  //   {
  //     Point pta;
  //     if (edge_face_collision(geom_0.point(faces_0[f][0]),
  // 			      geom_0.point(faces_0[f][1]),
  // 			      geom_0.point(faces_0[f][2]),
  // 			      geom_1.point(edges_1[e][0]),
  // 			      geom_1.point(edges_1[e][1]),
  // 			      pta)) 
  // 	points.push_back(pta);
	
  //     Point ptb;
  //     if (edge_face_collision(geom_1.point(faces_1[f][0]),
  // 			      geom_1.point(faces_1[f][1]),
  // 			      geom_1.point(faces_1[f][2]),
  // 			      geom_0.point(edges_0[e][0]),
  // 			      geom_0.point(edges_0[e][1]),
  // 			      ptb)) 
  // 	points.push_back(ptb);
  //   }
  // }    
  
  // // Edge edge intersection (only needed in very few cases)
  // for (int i=0; i<6; ++i) 
  // {
  //   for (int j=0; j<6; ++j) 
  //   {
  //     Point pt;
  //     if (collides_edge_edge(geom_0.point(edges_0[i][0]),
  // 			     geom_0.point(edges_0[i][1]),
  // 			     geom_1.point(edges_1[j][0]),
  // 			     geom_1.point(edges_1[j][1]))) 
  //     { 
	
  // 	points.push_back(pt);
  //     }
  //   }
  // }
  
  // // Remove duplicate nodes
  // std::vector<Point> tmp; 
  // tmp.reserve(points.size());
  // for (std::size_t i=0; i<points.size(); ++i) 
  // {
  //   bool different=true;
  //   for (std::size_t j=i+1; j<points.size(); ++j) 
  //   {
  //     if ((points[i]-points[j]).norm()<same_point_tol) {
  // 	different=false;
  // 	break;
  //     }
  //   }

  //   if (different) tmp.push_back(points[i]);
  // }
  // points=tmp;

  // // We didn't find sufficiently many points: can't form any
  // // tetrahedra.
  // if (points.size()<4) return std::vector<double>();

  // // Points forming the tetrahedral partitioning of the polyhedron. We
  // // have 4 points per tetrahedron in three dimensions => 12 doubles
  // // per tetrahedron.
  // std::vector<double> triangulation;

  // // Start forming a tesselation
  // if (points.size()==4) 
  // {
  //   // Include if determinant is sufficiently large. The determinant
  //   // can possibly be computed in a more stable way if needed.
  //   const double det=(points[3]-points[0]).dot((points[1]-points[0]).cross(points[2]-points[0]));
  //   if (std::abs(det)>tet_det_tol) 
  //   {
  //     if (det<-tet_det_tol) std::swap(points[0],points[1]);
	  
  //     // One tet with four vertices in 3D gives 12 doubles
  //     triangulation.resize(12); 
  //     for (std::size_t m=0,idx=0; m<4; ++m) 
  // 	for (std::size_t d=0; d<3; ++d,++idx) 
  // 	  triangulation[idx]=points[m][d];
	  
  //   }
  //   // Note: this can be empty if the tetrahedron was not sufficiently
  //   // large
  //   return triangulation;
  // }

  // // Tetrahedra are created using the facet points and a center point.
  // Point polyhedroncenter=points[0];
  // for (std::size_t i=1; i<points.size(); ++i) 
  //   polyhedroncenter+=points[i];
  // polyhedroncenter/=points.size();

  // // Data structure for storing checked triangle indices (do this
  // // better with some fancy stl structure?)
  // const int N=points.size(), N2=points.size()*points.size();
  // std::vector<int> checked(N*N2+N2+N,0);

  // // Find coplanar points
  // for (std::size_t i=0; i<points.size(); ++i) 
  // {
  //   for (std::size_t j=i+1; j<points.size(); ++j) 
  //   {
  //     for (std::size_t k=0; k<points.size(); ++k) 
  //     {
  // 	if (checked[i*N2+j*N+k]==0 and k!=i and k!=j) 
  // 	{
  // 	  // Check that triangle area is sufficiently large
  // 	  Point n=(points[j]-points[i]).cross(points[k]-points[i]);
  // 	  const double tridet=n.norm();
  // 	  if (tridet<tri_det_tol) { break; }
		  
  // 	  // Normalize normal
  // 	  n/=tridet; 
		  
  // 	  // Compute triangle center
  // 	  const Point tricenter=(points[i]+points[j]+points[k])/3.;
		  
  // 	  // Check whether all other points are on one side of thus
  // 	  // facet. Initialize as true for the case of only three
  // 	  // coplanar points.
  // 	  bool on_convex_hull=true; 
		  
  // 	  // Compute dot products to check which side of the plane
  // 	  // (i,j,k) we're on. Note: it seems to be better to compute
  // 	  // n.dot(points[m]-n.dot(tricenter) rather than
  // 	  // n.dot(points[m]-tricenter).
  // 	  std::vector<double> ip(points.size(),-(n.dot(tricenter)));
  // 	  for (std::size_t m=0; m<points.size(); ++m) 
  // 	    ip[m]+=n.dot(points[m]);
		  
  // 	  // Check inner products range by finding max & min (this
  // 	  // seemed possibly more numerically stable than checking all
  // 	  // vs all and then break).
  // 	  double minip=9e99, maxip=-9e99;
  // 	  for (size_t m=0; m<points.size(); ++m) 
  // 	    if (m!=i and m!=j and m!=k)
  // 	    {
  // 	      minip = (minip>ip[m]) ? ip[m] : minip;
  // 	      maxip = (maxip<ip[m]) ? ip[m] : maxip;
  // 	    }

  // 	  // Different sign => triangle is not on the convex hull
  // 	  if (minip*maxip<-DOLFIN_EPS) 
  // 	    on_convex_hull=false;
		  
  // 	  if (on_convex_hull) 
  // 	  {
  // 	    // Find all coplanar points on this facet given the
  // 	    // tolerance coplanar_tol
  // 	    std::vector<std::size_t> coplanar; 
  // 	    for (std::size_t m=0; m<points.size(); ++m) 
  // 	      if (std::abs(ip[m])<coplanar_tol) 
  // 		coplanar.push_back(m);
		      
  // 	    // Mark this plane (how to do this better?)
  // 	    for (std::size_t m=0; m<coplanar.size(); ++m) 
  // 	      for (std::size_t n=m+1; n<coplanar.size(); ++n) 
  // 		for (std::size_t o=n+1; o<coplanar.size(); ++o) 
  // 		  checked[coplanar[m]*N2+coplanar[n]*N+coplanar[o]]=
  // 		    checked[coplanar[m]*N2+coplanar[o]*N+coplanar[n]]=
  // 		    checked[coplanar[n]*N2+coplanar[m]*N+coplanar[o]]=
  // 		    checked[coplanar[n]*N2+coplanar[o]*N+coplanar[m]]=
  // 		    checked[coplanar[o]*N2+coplanar[n]*N+coplanar[m]]=
  // 		    checked[coplanar[o]*N2+coplanar[m]*N+coplanar[n]]=1;
		      
  // 	    // Do the actual tesselation using the coplanar points and
  // 	    // a center point
  // 	    if (coplanar.size()==3) 
  // 	    { 
  // 	      // Form one tetrahedron
  // 	      std::vector<Point> cand(4);
  // 	      cand[0]=points[coplanar[0]];
  // 	      cand[1]=points[coplanar[1]];
  // 	      cand[2]=points[coplanar[2]];
  // 	      cand[3]=polyhedroncenter;
			  
  // 	      // Include if determinant is sufficiently large
  // 	      const double det=(cand[3]-cand[0]).dot((cand[1]-cand[0]).cross(cand[2]-cand[0]));
  // 	      if (std::abs(det)>tet_det_tol) 
  // 	      {
  // 		if (det<-tet_det_tol) 
  // 		  std::swap(cand[0],cand[1]);

  // 		for (std::size_t m=0; m<4; ++m) 
  // 		  for (std::size_t d=0; d<3; ++d) 
  // 		    triangulation.push_back(cand[m][d]);
  // 	      }

  // 	    }
  // 	    else 
  // 	    {
  // 	      // Tesselate as in the triangle-triangle intersection
  // 	      // case: First sort points using a Graham scan, then
  // 	      // connect to form triangles. Finally form tetrahedra
  // 	      // using the center of the polyhedron.
	      
  // 	      // Use the center of the coplanar points and point no 0
  // 	      // as reference for the angle calculation
  // 	      Point pointscenter=points[coplanar[0]];
  // 	      for (std::size_t m=1; m<coplanar.size(); ++m) 
  // 		pointscenter+=points[coplanar[m]];
  // 	      pointscenter/=coplanar.size();
			  
  // 	      std::vector<std::pair<double, std::size_t> > order;
  // 	      Point ref=points[coplanar[0]]-pointscenter;
  // 	      ref/=ref.norm();
  // 	      const Point orthref=n.cross(ref);
			  
  // 	      // Calculate and store angles
  // 	      for (std::size_t m=1; m<coplanar.size(); ++m) 
  // 	      {		
  // 		const Point v=points[coplanar[m]]-pointscenter;
  // 		const double frac=ref.dot(v)/v.norm();
  // 		double alpha;
  // 		if (frac<=-1) alpha=DOLFIN_PI;
  // 		else if (frac>=1) alpha=0;
  // 		else { 
  // 		  alpha=acos(frac);
  // 		  if (v.dot(orthref)<0) 
  // 		    alpha=2*DOLFIN_PI-alpha; 
  // 		}
  // 		order.push_back(std::make_pair(alpha,m));
  // 	      }

  // 	      // Sort angles
  // 	      std::sort(order.begin(),order.end());
			  
  // 	      // Tesselate
  // 	      for (std::size_t m=0; m<coplanar.size()-2; ++m) 
  // 	      {
  // 		// Candidate tetrahedron:
  // 		std::vector<Point> cand(4);
  // 		cand[0]=points[coplanar[0]];
  // 		cand[1]=points[coplanar[order[m].second]];
  // 		cand[2]=points[coplanar[order[m+1].second]];
  // 		cand[3]=polyhedroncenter;

  // 		// Include tetrahedron if determinant is "large"
  // 		const double det=(cand[3]-cand[0]).dot((cand[1]-cand[0]).cross(cand[2]-cand[0]));
  // 		if (std::abs(det)>tet_det_tol) 
  // 		{
  // 		  if (det<-tet_det_tol) 
  // 		    std::swap(cand[0],cand[1]);
  // 		  for (std::size_t n=0; n<4; ++n) 
  // 		    for (std::size_t d=0; d<3; ++d) 
  // 		      triangulation.push_back(cand[n][d]);
  // 		}
  // 	      }
  // 	    }
  // 	  }
  // 	}
  //     }
  //   }
  // }



  return triangulation;
}
//-----------------------------------------------------------------------------

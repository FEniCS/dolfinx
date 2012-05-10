#ifndef __CSG_CGAL_MESH_GENERATOR2D_H
#define __CSG_CGAL_MESH_GENERATOR2D_H


namespace dolfin
{

  // Forward declarations
  class Mesh;
  class CSGGeometry;

  /// Mesh generator for Constructive Solid Geometry (CSG)
  /// utilizing CGALs boolean operation on Nef_polyhedrons.

  class CSGCGALMeshGenerator2D
  {
  public :
    CSGCGALMeshGenerator2D(const CSGGeometry& geometry);
    ~CSGCGALMeshGenerator2D();
    void generate(Mesh& mesh);

    //TODO: Add meshing parameters
  private:
    const CSGGeometry& geometry;
  };

}

#endif

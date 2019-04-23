#pragma once

namespace dolfin
{
/*! \namespace dolfin::mesh
    \brief Mesh data structures

    Representations of meshes and support for operations on meshes.
*/
}

// DOLFIN mesh interface

#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/Edge.h>
#include <dolfin/mesh/Face.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Connectivity.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshGeometry.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/MeshPartitioning.h>
#include <dolfin/mesh/MeshQuality.h>
#include <dolfin/mesh/Topology.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <dolfin/mesh/PeriodicBoundaryComputation.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/mesh/Vertex.h>

#pragma once

namespace dolfinx
{
/*! \namespace dolfinx::mesh
    \brief Mesh data structures

    Representations of meshes and support for operations on meshes.
*/
}

// DOLFINX mesh interface

#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshEntity.h>
#include <dolfinx/mesh/MeshFunction.h>
#include <dolfinx/mesh/MeshIterator.h>
#include <dolfinx/mesh/MeshQuality.h>
#include <dolfinx/mesh/MeshValueCollection.h>
#include <dolfinx/mesh/Partitioning.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>

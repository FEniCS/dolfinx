#pragma once

namespace dolfin
{
/*! \namespace dolfin::graph
    \brief Graph data structures and algorithms

    Data structures for building and representing graphs, and algorithms on
   graphs, e.g., re-ordering and partitioning.
*/
}

// DOLFIN graph interface

#include <dolfin/graph/BoostGraphOrdering.h>
#include <dolfin/graph/Graph.h>
#include <dolfin/graph/GraphBuilder.h>
#include <dolfin/graph/SCOTCH.h>

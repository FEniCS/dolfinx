#include <catch2/catch_test_macros.hpp>

#include <mpi.h>

#include "dolfinx/mesh/generation.h"

using namespace dolfinx;

TEST_CASE("Generation", "create_interval")
{
    mesh::Mesh<double> mesh = mesh::create_interval(MPI_COMM_SELF, 4, {0., 1.});

    {
        int comp_result;   
        MPI_Comm_compare(mesh.comm(), MPI_COMM_SELF, &comp_result);
        CHECK(comp_result == MPI_CONGRUENT);
    }

    CHECK(mesh.geometry().dim() == 1);
    auto vertices = mesh.geometry().x();
    CHECK(vertices[0*3] == 0.0);
    CHECK(vertices[1*3] == 0.25);
    CHECK(vertices[2*3] == 0.5);
    CHECK(vertices[3*3] == 0.75);
    CHECK(vertices[4*3] == 1.0);

    mesh.topology()->create_connectivity(0, 1);
    auto point_conn = mesh.topology()->connectivity(0, 1);

    CHECK(point_conn->num_nodes() == 5);

    CHECK(point_conn->num_links(0) == 1);
    CHECK(point_conn->num_links(1) == 2);
    CHECK(point_conn->num_links(2) == 2);
    CHECK(point_conn->num_links(3) == 2);
    CHECK(point_conn->num_links(4) == 1);

    CHECK(point_conn->links(0)[0] == 0);
    CHECK(point_conn->links(1)[0] == 0);
    CHECK(point_conn->links(1)[1] == 1);
    CHECK(point_conn->links(2)[0] == 1);
    CHECK(point_conn->links(2)[1] == 2);
    CHECK(point_conn->links(3)[0] == 2);
    CHECK(point_conn->links(3)[1] == 3);
    CHECK(point_conn->links(4)[0] == 3);
}

// TODO extend for further mesh types.

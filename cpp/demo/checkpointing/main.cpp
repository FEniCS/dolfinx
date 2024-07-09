// # Checkpointing
//

#include <dolfinx.h>
#include <mpi.h>

using namespace dolfinx;

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);

  // {
  //   int i=0;
  //   while (i == 0)
  //     sleep(5);
  // }

  // Create mesh and function space
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
  auto mesh = std::make_shared<mesh::Mesh<float>>(mesh::create_rectangle<float>(
      MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {4, 4},
      mesh::CellType::quadrilateral, part));

  auto writer = io::ADIOS2Engine(mesh->comm(), "mesh.bp", "mesh-write", "BP5",
                                 adios2::Mode::Write);

  // auto io = writer.io();
  // auto engine = writer.engine();

  // io::checkpointing::write(io, engine, mesh);
  io::checkpointing::write(mesh->comm(), "mesh.bp", "mesh-write", mesh);

  MPI_Finalize();
  return 0;
}

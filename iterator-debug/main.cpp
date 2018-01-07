#include <dolfin.h>
#include <boost/timer/timer.hpp>

using namespace dolfin;

int main()
{
  //std::array<Point, 2> pt = {Point(0.,0.), Point(1.,1.)};
  //auto mesh = std::make_shared<Mesh>(RectangleMesh::create(MPI_COMM_WORLD, pt, {{320, 320}}, CellType::Type::triangle));
  std::array<Point, 2> pt = {Point(0.,0.,0.), Point(1.,1.,1.)};
  auto mesh = std::make_shared<Mesh>(BoxMesh::create(MPI_COMM_WORLD, pt,
                                                    {{300, 200, 200}},
                                                     CellType::Type::tetrahedron));
  //auto mesh = std::make_shared<Mesh>(BoxMesh::create(MPI_COMM_WORLD, pt,
  //                                                   {{1, 1, 1}}, CellType::Type::tetrahedron));

  {
    Timer t0("old iterators");
    std::size_t p = 0;
    boost::timer::cpu_timer t;
    for (CellIterator c(*mesh); !c.end(); ++c)
    {
      //std::cout << "Start old v loop" << std::endl;
      for (VertexIterator v(*c); !v.end(); ++v)
      {
        //std::cout << "   vertex loop: " << v->index() << std::endl;
        p += v->index();
        //p += std::sqrt(p);
      }
      //p += c->index();
    }
    auto tend = t.elapsed();
    std::cout << p << std::endl;
    std::cout << boost::timer::format(tend) << "\n";
  }

  std::cout << "A--------------------------" << std::endl;
  {
    Timer t0("new (range-based)");
    std::size_t p = 0;
    boost::timer::cpu_timer t;
    //std::cout << "Start cell loop" << std::endl;
    for (const auto &c : mesh_entities(*mesh, mesh->topology().dim()))
    {
      //std::cout << "\n1. Create v list (range)" << std::endl;
      //const auto vert = entities<Vertex>(c, 0);
      //std::cout << "2. Start v loop (range)" << std::endl;
      //std::cout << "Start vertex loop" << std::endl;
      for (const auto& v : entities(c, 0))
      {
        //std::cout << "   vertex loop: " << v.index() << std::endl;
        p += v.index();
        //p += std::sqrt(p);
      }
      //p += c.index();
    }
    auto tend = t.elapsed();
    std::cout << p << std::endl;
    std::cout << boost::timer::format(tend) << "\n";
  }

  std::cout << "B--------------------------" << std::endl;
  {
    Timer t0("new (no range-based)");
    std::size_t p = 0;
    boost::timer::cpu_timer t;
    for (const auto &c : mesh_entities(*mesh, mesh->topology().dim()))
    {
      //std::cout << "\n1. Create v list (no range)" << std::endl;
      entities vert(c, 0);
      //std::cout << "2. Create begin/end (no range)" << std::endl;
      const auto v0 = vert.begin();
      const auto v1 = vert.end();
      //std::cout << "3. Start v loop (no range)" << std::endl;
      for (auto v = v0; v != v1; ++v)
      {
        p += v->index();
        //p += std::sqrt(p);
      }
      //p += c.index();
    }
    auto tend = t.elapsed();
    std::cout << p << std::endl;
    std::cout << boost::timer::format(tend) << "\n";
  }

  /*
  std::cout << "--------------------------" << std::endl;
  {
    Timer t0("new (outer) + old (inner)");
    std::size_t p = 0;
    boost::timer::cpu_timer t;
    for (auto &c : cells(*mesh))
    {
      // Use old iterator
      for (VertexIterator v(c); !v.end(); ++v)
      {
        p += v->index();
        //p += std::sqrt(p);
      }
    }
    auto tend = t.elapsed();
    std::cout << p << std::endl;
    std::cout << boost::timer::format(tend) << "\n";
  }
  */

  list_timings(TimingClear::clear, {{TimingType::wall}});
  return 0;
}

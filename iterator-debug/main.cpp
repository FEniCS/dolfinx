#include <dolfin.h>
#include <boost/timer/timer.hpp>

using namespace dolfin;

int main()
{
  //std::array<Point, 2> pt = {Point(0.,0.), Point(1.,1.)};
  //auto mesh = std::make_shared<Mesh>(RectangleMesh::create(MPI_COMM_WORLD, pt, {{320, 320}}, CellType::Type::triangle));
  std::array<Point, 2> pt = {Point(0.,0.,0.), Point(1.,1.,1.)};
  auto mesh = std::make_shared<Mesh>(BoxMesh::create(MPI_COMM_WORLD, pt,
                                                     {{200, 200, 200}}, CellType::Type::tetrahedron));

  {
    Timer t0("old iterators");
    std::size_t p = 0;
    boost::timer::cpu_timer t;
    for (CellIterator c(*mesh); !c.end(); ++c)
    {
      for (VertexIterator v(*c); !v.end(); ++v)
        p += v->index();
      p += c->index();
    }
    auto tend = t.elapsed();
    std::cout << p << std::endl;
    std::cout << boost::timer::format(tend) << "\n";
  }

  {
    Timer t0("new");
    std::size_t p = 0;
    boost::timer::cpu_timer t;
    for (auto &c : cells(*mesh))
    {
      for (auto &v : vertices(c))
        p += v.index();
      p += c.index();
    }
    auto tend = t.elapsed();
    std::cout << p << std::endl;
    std::cout << boost::timer::format(tend) << "\n";
  }

  {
    Timer t0("new (outer) + old (inner)");
    std::size_t p = 0;
    boost::timer::cpu_timer t;
    for (auto &c : cells(*mesh))
    {
      // Use old iterator
      for (VertexIterator v(c); !v.end(); ++v)
        p += v->index();
    }
    auto tend = t.elapsed();
    std::cout << p << std::endl;
    std::cout << boost::timer::format(tend) << "\n";
  }

  /*
  {
    boost::timer::cpu_timer t;
    Timer t0("new2");
    std::size_t p = 0;

    for (auto &c : cells(*mesh))
    {
      // New with auto
      for (auto &v : vertices(c))
        p += v.index();
    }
    std::cout << p << std::endl;
    std::cout << boost::timer::format(t.elapsed()) << "\n";
  }

  {
    boost::timer::cpu_timer t;
    Timer t0("new3");
    std::size_t p = 0;

    for (auto &c : cells(*mesh))
    {
      // New explicit
      auto u = vertices(c);
      MeshIterator<Vertex> v0 = u.begin();
      MeshIterator<Vertex> v1 = u.end();
      for (MeshIterator<Vertex> v = v0; v != v1; ++v)
        p += v->index();
    }
    std::cout << p << std::endl;
    std::cout << boost::timer::format(t.elapsed()) << "\n";
  }
  */

  list_timings(TimingClear::clear, {{TimingType::wall}});
  return 0;
}

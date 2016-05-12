#include <cmath>
#include <fstream>
#include <dolfin.h>

#include </home/august/dev/fenics-dev/dolfin-multimesh/dolfin/geometry/dolfin_simplex_tools.h>

using namespace dolfin;

double compute_volume(const MultiMesh& multimesh,
		      double exact_volume)
{
  std::cout << "\n" << __FUNCTION__<< std::endl;

  double volume = 0;
  std::vector<double> all_volumes;

  std::ofstream file("quadrature_volume.txt");
  if (!file.good()) { std::cout << "file not good\n"; exit(0); }
  file.precision(20);

  // Sum contribution from all parts
  std::cout << "Sum contributions\n";
  for (std::size_t part = 0; part < multimesh.num_parts(); part++)
  {
    std::cout << "% part " << part;
    double part_volume = 0;
    std::vector<double> status(multimesh.part(part)->num_cells(), 0);

    // Uncut cell volume given by function volume
    const auto uncut_cells = multimesh.uncut_cells(part);
    for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
    {
      const Cell cell(*multimesh.part(part), *it);
      volume += cell.volume();
      //std::cout << std::setprecision(20) << cell.volume() <<'\n';
      part_volume += cell.volume();
      status[*it] = 1;
      //file << "0 0 "<< cell.volume() << '\n';
    }

    std::cout << "\t uncut volume "<< part_volume << ' ';

    // Cut cell volume given by quadrature rule
    const auto& cut_cells = multimesh.cut_cells(part);
    for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
    {
      const auto& qr = multimesh.quadrature_rule_cut_cell(part, *it);
      for (std::size_t i = 0; i < qr.second.size(); ++i)
      {
	file << qr.first[2*i]<<' '<<qr.first[2*i+1]<<' '<<qr.second[i]<<'\n';
	volume += qr.second[i];
	part_volume += qr.second[i];
	//std::cout << qr.first[2*i]<<' '<<qr.first[2*i+1]<<'\n';
      }
      status[*it] = 2;
    }
    std::cout << "\ttotal volume " << part_volume << std::endl;

    all_volumes.push_back(part_volume);

    tools::dolfin_write_medit_triangles("status",*multimesh.part(part),part,&status);
  }
  file.close();

  return volume;
}



double compute_interface_area(const MultiMesh& multimesh,
			      double exact_area)
{
  std::cout << "\n" << __FUNCTION__ << std::endl;

  double area = 0;
  std::vector<double> all_areas;

  std::ofstream file("quadrature_interface.txt");
  if (!file.good()) { std::cout << "file not good\n"; exit(0); }
  file.precision(20);

  // Sum contribution from all parts
  std::cout << "Sum contributions\n";
  for (std::size_t part = 0; part < multimesh.num_parts(); part++)
  {
    std::cout << "% part " << part << ' ';
    double part_area = 0;
    const auto& quadrature_rules = multimesh.quadrature_rule_interface(part);

    // // Uncut cell area given by function area
    // const auto uncut_cells = multimesh.uncut_cells(part);
    // for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
    // {
    //   const Cell cell(*multimesh.part(part), *it);
    //   area += cell.area();
    // 	//std::cout << std::setprecision(20) << cell.area() <<'\n';
    //   part_area += cell.area();
    // 	status[*it] = 1;
    // 	//file << "0 0 "<< cell.area() << '\n';
    // }

    // std::cout << "\t uncut area "<< part_area << ' ';


    // Get collision map
    const auto& cmap = multimesh.collision_map_cut_cells(part);
    for (auto it = cmap.begin(); it != cmap.end(); ++it)
    {
      const unsigned int cut_cell_index = it->first;
      const auto& cutting_cells = it->second;

      // Iterate over cutting cells
      for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
      {
	// Get quadrature rule for interface part defined by
	// intersection of the cut and cutting cells
	const std::size_t k = jt - cutting_cells.begin();
	// std::cout << cut_cell_index << ' ' << k <<' ' << std::flush
	// 	    << quadrature_rules.size() << ' '
	// 	    << quadrature_rules.at(cut_cell_index).size() << "   " << std::flush;
	dolfin_assert(k < quadrature_rules.at(cut_cell_index).size());
	const auto& qr = quadrature_rules.at(cut_cell_index)[k];
	std::stringstream ss;
	for (std::size_t i = 0; i < qr.second.size(); ++i)
	{
	  file << qr.first[2*i]<<' '<<qr.first[2*i+1]<<' '<<qr.second[i]<<'\n';
	  //std::cout << qr.second[i]<<' ';
	  area += qr.second[i];
	  part_area += qr.second[i];
	  //std::cout << qr.first[2*i]<<' '<<qr.first[2*i+1]<<'\n';
	}
	//std::cout << std::endl;
      }
    }
    std::cout << "total area " << part_area << std::endl;
    all_areas.push_back(part_area);
  }
  file.close();

  return area;
}


int main(int argc, char* argv[])
{
  const std::size_t N = 1;

  auto mesh_0 = std::make_shared<UnitSquareMesh>(N, N);
  //auto mesh_1 = std::make_shared<RectangleMesh>(Point(0.1, 0.1), Point(0.9, 0.9), N, N);
  auto mesh_1 = std::make_shared<RectangleMesh>(Point(0.1, 0.1), Point(0.9, 0.9), N, N);
  mesh_1->translate(Point(-0.05, 0.05));
  auto mesh_2 = std::make_shared<RectangleMesh>(Point(0.2, 0.2), Point(0.8, 0.8), N, N);
  // tools::dolfin_write_medit_triangles("mesh0",*mesh_0);
  // tools::dolfin_write_medit_triangles("mesh1",*mesh_1);
  // tools::dolfin_write_medit_triangles("mesh2",*mesh_2);

  double exact_volume = (0.9 - 0.1)*(0.9 - 0.1)*6; // for mesh_0 and mesh_1
  //exact_volume += (0.8 - 0.2)*(0.8 - 0.2)*6; // mesh_1 and mesh_2

  double exact_area = 4*0.9;


  // Build multimesh
  auto multimesh = std::make_shared<MultiMesh>();
  multimesh->add(mesh_0);
  multimesh->add(mesh_1);
  //multimesh->add(mesh_2);
  multimesh->build(6); // qr generated here

  double volume = compute_volume(*multimesh, 0);
  double area = compute_interface_area(*multimesh, 0);
  std::cout << "volume " << volume << ' ' << exact_volume <<' '<< std::abs(volume-exact_volume) << std::endl
	    << "area " << area << ' ' << exact_area << ' '<<std::abs(area-exact_area) << std::endl;



}

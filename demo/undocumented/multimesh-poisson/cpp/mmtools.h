#include <dolfin.h>

#include <dolfin/geometry/dolfin_simplex_tools.h>

namespace mmtools
{
  using namespace dolfin;

  //------------------------------------------------------------------------------
  void writemarkers(std::size_t step,
		    const MultiMesh& mm)
  {
    for (std::size_t part = 0; part < mm.num_parts(); ++part)
    {
      std::stringstream ss; ss << part;
      const std::size_t n = mm.part(part)->num_cells();
      std::vector<int> uncut(n, -1), cut(n, -1), covered(n, -1);
      for (const auto c: mm.uncut_cells(part)) uncut[c] = 0;
      for (const auto c: mm.cut_cells(part)) cut[c] = 1;
      for (const auto c: mm.covered_cells(part)) covered[c] = 2;
      tools::dolfin_write_medit_triangles("uncut"+ss.str(),*mm.part(part),step,&uncut);
      tools::dolfin_write_medit_triangles("cut"+ss.str(),*mm.part(part),step,&cut);
      tools::dolfin_write_medit_triangles("covered"+ss.str(),*mm.part(part),step,&covered);
    }
    tools::dolfin_write_medit_triangles("multimesh",mm,step);

  }

  //------------------------------------------------------------------------------
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


  //------------------------------------------------------------------------------
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


  //------------------------------------------------------------------------------
  void plot_normals(const MultiMesh& multimesh)
  {
    std::cout << "\n" << __FUNCTION__ << std::endl;
    const std::vector<std::string> colors = {{ "'b'", "'g'", "'r'" }};
    const std::vector<std::string> marker = {{ "'.'", "'o'", "'x'" }};

    //for (std::size_t part = 0; part < multimesh.num_parts(); part++)
    const std::size_t part = 1;
    {
      std::cout << "% part " << part << ' ' <<std::endl;
      const auto& cmap = multimesh.collision_map_cut_cells(part);
      const auto& quadrature_rules = multimesh.quadrature_rule_interface(part);
      const auto& normals = multimesh.facet_normals(part);

      for (auto it = cmap.begin(); it != cmap.end(); ++it)
      {
	const unsigned int cut_cell_index = it->first;
	const auto& cutting_cells = it->second;

	const Cell cut_cell(*multimesh.part(part), cut_cell_index);
	std::cout << tools::drawtriangle(cut_cell, colors[part]);

	// Iterate over cutting cells
	for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
	{
	  const std::size_t cutting_cell_part = jt->first;

	  if (cutting_cell_part == 2)
	  {
	    const Cell cutting_cell(*multimesh.part(cutting_cell_part), jt->second);
	    std::cout << tools::drawtriangle(cutting_cell, colors[cutting_cell_part]);

	    // Get quadrature rule for interface part defined by
	    // intersection of the cut and cutting cells
	    const std::size_t k = jt - cutting_cells.begin();
	    const auto& qr = quadrature_rules.at(cut_cell_index)[k];
	    const auto& nn = normals.at(cut_cell_index)[k];

	    for (std::size_t i = 0; i < qr.second.size(); ++i)
	    {
	      const Point p(qr.first[2*i], qr.first[2*i+1]);
	      std::cout << tools::plot(p,"'k.'");
	      const Point n(nn[2*i], nn[2*i+1]);
	      const double d = 0.01;
	      std::cout << tools::drawarrow(p, p+d*n, colors[cutting_cell_part]);
	    }
	  }
	}
      }

      // for (const auto cell_no: multimesh.cut_cells(part))
      // {
      //   const auto qrmap = multimesh.quadrature_rule_interface(part).find(cell_no);
      //   const std::vector<quadrature_rule> qr = qrmap->second;

      //   const auto fnmap = multimesh.facet_normals(part).find(cell_no);
      //   const std::vector<std::vector<double>> normals = fnmap->second;

      //   //std::cout << qr.size() << ' ' << normals.size() << std::endl;
      //   dolfin_assert(qr.size() == normals.size());

      //   for (std::size_t i = 0; i < qr.size(); ++i)
      //   {
      // 	for (std::size_t j = 0; j < qr[i].second.size(); ++j)
      // 	{
      // 	  const Point p(qr[i].first[2*j], qr[i].first[2*j+1]);
      // 	  std::cout << tools::plot(p,"'k.'");
      // 	  const Point n(normals[i][2*j],normals[i][2*j+1]);
      // 	  const double d = 0.01;
      // 	  std::cout << tools::drawarrow(p, p+d*n);
      // 	}
      // 	std::cout << std::endl;
      //   }
      // }

    }
  }


  //------------------------------------------------------------------------------
  void evaluate_at_qr(const MultiMesh& mm,
		      const MultiMeshFunction& uh)
  {
    std::cout << __FUNCTION__ << std::endl;

    for (std::size_t part = 0; part < mm.num_parts(); ++part)
    {
      std::cout << "\npart " << part << '\n';

      // get vertex values
      std::vector<double> vertex_values;
      uh.part(part)->compute_vertex_values(vertex_values, *mm.part(part));

      const std::vector<std::string> colors = {{ "'b'", "'g'", "'r'" }};
      std::vector<std::size_t> cells;

      // cells colliding with the cut cells
      const auto collision_map = mm.collision_map_cut_cells(part);

      // loop over cut cells
      for (const auto cell_no: mm.cut_cells(part))
      {
	// all qr on cell_no
	const auto qr = mm.quadrature_rule_cut_cell(part, cell_no);

	// loop over qr
	for (std::size_t i = 0; i < qr.second.size(); ++i)
	{
	  const Point p(qr.first[2*i], qr.first[2*i+1]);
	  const double uhval = (*uh.part(part))(p.x(), p.y());

	  // if evaluated function big...
	  if (std::abs(uhval) > 1)
	  {
	    // save cell no
	    cells.push_back(cell_no);
	    const std::string color = qr.second[i] > 0 ? "'.'" : "'x'";
	    std::cout << tools::matlabplot(p,color) <<" % " << qr.second[i] << ' '
		      << /*std::setprecision(15) <<*/ uhval << " (";

	    // print nodal uh values
	    const Cell cell(*mm.part(part), cell_no);
	    for (std::size_t j = 0; j < cell.num_vertices(); ++j)
	      std::cout << cell.entities(0)[j] << ' '<<vertex_values[cell.entities(0)[j]] <<' ';
	    std::cout << ")\n";
	  }
	}
      }

      // make cell numbers unique
      std::sort(cells.begin(), cells.end());
      const auto new_end = std::unique(cells.begin(), cells.end());
      cells.erase(new_end, cells.end());

      // loop over all cells with large uh values
      for (const auto cell_no: cells)
      {
	std::cout << "% cell with large uh:\n";
	const Cell cell(*mm.part(part), cell_no);
	std::cout << tools::drawtriangle(cell);

	// compute net weight (~visible area)
	const auto qr = mm.quadrature_rule_cut_cell(part, cell_no);
	double net_weight = 0;
	std::cout << " % ";
	for (const auto w: qr.second)
	{
	  net_weight += w;
	  std::cout << ' '<<w;
	}
	std::cout << "\n% net weight = " << net_weight << '\n';

	// also display all colliding cells
	const auto it = collision_map.find(cell_no);
	dolfin_assert(it->first == cell_no);
	std::cout << "% colliding:\n";
	for (const auto cpair: it->second)
	{
	  const Cell cutting_cell(*mm.part(cpair.first), cpair.second);
	  std::cout << tools::drawtriangle(cutting_cell,colors[cpair.first]);
	}
      }

    }
    PPause;
  }

  //------------------------------------------------------------------------------
  void find_max(std::size_t step,
		const MultiMesh& multimesh,
		const MultiMeshFunction& u,
		File& uncut0_file, File& uncut1_file, File& uncut2_file,
		File& cut0_file, File& cut1_file, File& cut2_file,
		File& covered0_file, File& covered1_file, File& covered2_file)

  {
    std::cout << "\tSolution: max min step " << step <<' ' << u.vector()->max() << ' ' << u.vector()->min() << '\n';

    for (std::size_t part = 0; part < multimesh.num_parts(); ++part)
    {
      // get max on vertex values
      std::vector<double> vertex_values;
      u.part(part)->compute_vertex_values(vertex_values,
					  *multimesh.part(part));
      const double maxvv = *std::max_element(vertex_values.begin(),
					     vertex_values.end());

      // get max on uncut, cut and covered
      const std::vector<std::vector<unsigned int>> cells
	= {{ multimesh.uncut_cells(part),
	     multimesh.cut_cells(part),
	     multimesh.covered_cells(part) }};
      const std::vector<std::string> type = {{ "uncut", "cut", "covered" }};
      std::vector<double> maxvals(cells.size(), 0);

      for (std::size_t k = 0; k < cells.size(); ++k)
      {
	std::cout << "part " << part << " "<<k << ' '<<type[k]<< std::endl;
	if (cells[k].size())
	{
	  // Create meshfunction using markers
	  auto mesh_part = std::make_shared<Mesh>(*multimesh.part(part));
	  auto foo = std::make_shared<MeshFunction<std::size_t> >(mesh_part, mesh_part->topology().dim());
	  foo->set_all(0); // dummy
	  for (const auto cell: cells[k])
	    foo->set_value(cell, k+1);

	  // Create submesh out of meshfunction
	  auto sm = std::make_shared<SubMesh>(*multimesh.part(part), *foo, k+1);

	  // Interpolate on submesh
	  auto V = std::make_shared<P1::FunctionSpace>(sm);
	  auto usm = std::make_shared<Function>(V);

	  // test
	  usm->set_allow_extrapolation(true);

	  usm->interpolate(*u.part(part));

	  // Get max values on submesh
	  std::vector<double> vertex_values;
	  usm->compute_vertex_values(vertex_values);
	  maxvals[k] = *std::max_element(vertex_values.begin(), vertex_values.end());

	  // if (part == 0)
	  //   if (k == 0 or k == 1) {
	  //     std::cout << k <<'\n';
	  //     for (const auto cell: cells[k])
	  // 	std::cout << cell << ' ';
	  //     std::cout << '\n';
	  //   }

	  // if (marker == 1 and part == 0) {
	  //   for (const auto v: vertex_values)
	  //     std::cout << v<<' ';
	  //   std::cout << '\n';
	  // }

	  // save
	  switch(k) {
	  case 0: { // uncut
	    if (part == 0) uncut0_file << (*usm);
	    else if (part == 1) uncut1_file << (*usm);
	    else if (part == 2) uncut2_file << (*usm);
	    break;
	  }
	  case 1: { // cut
	    if (part == 0) cut0_file << (*usm);
	    else if (part == 1) cut1_file << (*usm);
	    else if (part == 2) cut2_file << (*usm);
	    break;
	  }
	  case 2: { // covered
	    if (part == 0) covered0_file << (*usm);
	    else if (part == 1) covered1_file << (*usm);
	    else if (part == 2) covered2_file << (*usm);
	  }
	  }
	}
      }

      std::cout << "\tpart " << part
		<< " step " << step
		<< " all vertices " << maxvv
		<< " uncut " << maxvals[0]
		<< " cut " << maxvals[1]
		<< " covered " << maxvals[2] << '\n';

      if (maxvals[0] < 1) { exit(0); }
    }

  }
  //------------------------------------------------------------------------------

}

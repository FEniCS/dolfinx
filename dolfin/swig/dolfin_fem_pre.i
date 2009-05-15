// Ignore the const Mesh constructor of DofMap
%ignore dolfin::DofMap::DofMap(ufc::dof_map &,const Mesh &);
%ignore dolfin::DofMap::DofMap(boost::shared_ptr<ufc::dof_map>, const Mesh &);

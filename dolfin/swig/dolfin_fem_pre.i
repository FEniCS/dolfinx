// Ignore all but one shared_ptr Constructor to DofMap
%ignore dolfin::DofMap::DofMap(ufc::dof_map &,Mesh &);
%ignore dolfin::DofMap::DofMap(ufc::dof_map &,const Mesh &);
%ignore dolfin::DofMap::DofMap(boost::shared_ptr<ufc::dof_map>, const Mesh &);

// Ignore all but one shared_ptr Constructor to FiniteElement
%ignore dolfin::FiniteElement::FiniteElement(const ufc::finite_element&);

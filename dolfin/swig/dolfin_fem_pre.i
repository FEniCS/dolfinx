// Ignore reference (to FunctionSpaces and Functions) constructors of BoundaryConditions
%ignore dolfin::BoundaryCondition::BoundaryCondition(const FunctionSpace&);
%ignore dolfin::EqualityBC::EqualityBC(const FunctionSpace&, const SubDomain&);
%ignore dolfin::EqualityBC::EqualityBC(const FunctionSpace&, uint);
%ignore dolfin::DirichletBC::DirichletBC(const FunctionSpace&, 
					 const Function&, 
					 const SubDomain&,
					 std::string method="topological");

%ignore dolfin::DirichletBC::DirichletBC(const FunctionSpace&, 
					 const Function&,
					 const MeshFunction<uint>&, 
					 uint,
					 std::string method="topological");

%ignore dolfin::DirichletBC::DirichletBC(const FunctionSpace&, 
					 const Function&, 
					 uint,
					 std::string method="topological");

%ignore dolfin::PeriodicBC::PeriodicBC(const FunctionSpace&, const SubDomain&);

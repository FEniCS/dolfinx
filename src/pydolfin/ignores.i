// Things to ignore for PyDOLFIN

%ignore *::operator=;
%ignore *::operator[];
%ignore *::operator++;
%ignore operator<<;
%ignore operator int;
%ignore operator dolfin::uint;
%ignore operator dolfin::real;
%ignore operator std::string;
%ignore operator bool;

%ignore dolfin::Mesh::partition(dolfin::uint num_partitions, dolfin::MeshFunction<dolfin::uint>& partitions);
%ignore dolfin::dolfin_info_aptr;
%ignore operator<< <Mat>;
%ignore dolfin::Parameter;
%ignore dolfin::Parametrized;
%ignore dolfin::LogStream;
%ignore dolfin::ElementLibrary::create_finite_element(char const *);
%ignore dolfin::ElementLibrary::create_dof_map(char const *);
%ignore dolfin::MeshGeometry::x(uint n, uint i) const;
%ignore dolfin::uBlasVector::operator ()(uint i) const;
%ignore dolfin::cout;
%ignore dolfin::endl;

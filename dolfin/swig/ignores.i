// Things to ignore for PyDOLFIN
// TODO: Clean up this mess :)

%ignore *::operator=;
%ignore *::operator[];
%ignore *::operator++;
%ignore operator<<;
%ignore operator dolfin::uint;
%ignore operator std::string;
%ignore operator bool;

%ignore dolfin::Mesh::partition(dolfin::uint num_partitions, dolfin::MeshFunction<dolfin::uint>& partitions);
%ignore dolfin::dolfin_info_aptr;
//%ignore operator<< <Mat>;
%ignore dolfin::Parameter;
%ignore dolfin::Parametrized::get;
%ignore dolfin::Parametrized::set;
%ignore dolfin::Parametrized::add;
%ignore dolfin::LogStream;
%ignore dolfin::ElementLibrary::create_finite_element(char const *);
%ignore dolfin::ElementLibrary::create_dof_map(char const *);
%ignore dolfin::MeshGeometry::x(uint n, uint i) const;
%ignore dolfin::uBLASVector::operator ()(uint i) const;
%ignore dolfin::cout;
%ignore dolfin::endl;
%ignore *::operator<<(unsigned int);
%ignore dolfin::MeshConnectivity::set(uint entity, uint* connections);

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

%ignore operator<< <Mat>;
%ignore dolfin::Parameter;
%ignore dolfin::Parametrized;
%ignore dolfin::MeshGeometry::x(uint n, uint i) const;
%ignore dolfin::uBlasVector::operator ()(uint i) const;

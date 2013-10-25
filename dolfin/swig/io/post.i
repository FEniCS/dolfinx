// Instantiate io template functions

// Input
%template(__rshift__) dolfin::File::operator>> <GenericVector>;
%template(__rshift__) dolfin::File::operator>> <GenericMatrix>;
%template(__rshift__) dolfin::File::operator>> <Mesh>;
%template(__rshift__) dolfin::File::operator>> <LocalMeshData>;
%template(__rshift__) dolfin::File::operator>> <MeshFunction<int> >;
%template(__rshift__) dolfin::File::operator>> <MeshFunction<std::size_t> >;
%template(__rshift__) dolfin::File::operator>> <MeshFunction<double> >;
%template(__rshift__) dolfin::File::operator>> <MeshFunction<bool> >;
%template(__rshift__) dolfin::File::operator>> <MeshValueCollection<int> >;
%template(__rshift__) dolfin::File::operator>> <MeshValueCollection<std::size_t> >;
%template(__rshift__) dolfin::File::operator>> <MeshValueCollection<double> >;
%template(__rshift__) dolfin::File::operator>> <MeshValueCollection<bool> >;
%template(__rshift__) dolfin::File::operator>> <Parameters>;
%template(__rshift__) dolfin::File::operator>> <Function>;

// Output
%template(__lshift__) dolfin::File::operator<< <GenericVector>;
%template(__lshift__) dolfin::File::operator<< <GenericMatrix>;
%template(__lshift__) dolfin::File::operator<< <Mesh>;
%template(__lshift__) dolfin::File::operator<< <LocalMeshData>;
%template(__lshift__) dolfin::File::operator<< <MeshFunction<int> >;
%template(__lshift__) dolfin::File::operator<< <MeshFunction<std::size_t> >;
%template(__lshift__) dolfin::File::operator<< <MeshFunction<double> >;
%template(__lshift__) dolfin::File::operator<< <MeshFunction<bool> >;
%template(__lshift__) dolfin::File::operator<< <MeshValueCollection<int> >;
%template(__lshift__) dolfin::File::operator<< <MeshValueCollection<std::size_t> >;
%template(__lshift__) dolfin::File::operator<< <MeshValueCollection<double> >;
%template(__lshift__) dolfin::File::operator<< <MeshValueCollection<bool> >;
%template(__lshift__) dolfin::File::operator<< <Parameters>;
%template(__lshift__) dolfin::File::operator<< <Function>;

%extend dolfin::HDF5Attribute {
  void __setitem__(std::string key, double value) 
  { $self->set(key, value); }
  void __setitem__(std::string key, std::size_t value) 
  { $self->set(key, value); }
  void __setitem__(std::string key, std::string value) 
  { $self->set(key, value); }
  void __setitem__(std::string key, const std::vector<double>& value) 
  { $self->set(key, value); }
  //  void __setitem__(std::string key, const std::vector<std::size_t>& value) 
  //  { $self->set(key, value); }

%pythoncode %{
def __getitem__(self, key):
    attr_type = self.type_str(key)
    if attr_type=="string":
        return self.str(key)
    elif attr_type=="float":
        return float(self.str(key))
    elif attr_type=="int":
        return int(self.str(key))
    elif attr_type=="vectorfloat":
        return [float(x) for x in self.str(key).split(",")]
    elif attr_type=="vectorint":
        return [int(x) for x in self.str(key).split(",")]
    return None
%}

}


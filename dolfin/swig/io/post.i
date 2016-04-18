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
%template(__rshift__) dolfin::File::operator>> <Table>;
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
%template(__lshift__) dolfin::File::operator<< <Table>;
%template(__lshift__) dolfin::File::operator<< <Function>;

%extend dolfin::File {
%pythoncode %{
def __enter__(self) :
    return self

def __exit__(self, type, value, traceback) :
    pass
    # Do Nothing...
    #self.close()
%}
}

#ifdef HAS_HDF5
%extend dolfin::HDF5Attribute {
  void __setitem__(std::string key, double value)
  { $self->set(key, value); }
  void __setitem__(std::string key, std::size_t value)
  { $self->set(key, value); }
  void __setitem__(std::string key, std::string value)
  { $self->set(key, value); }
  void __setitem__(std::string key, const std::vector<double>& value)
  { $self->set(key, value); }
  void __setitem__(std::string key, const std::vector<std::size_t>& value)
  { $self->set(key, value); }

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

def __contains__(self, key):
    return self.exists(key)

def __len__(self, key):
    return len(self.list_attributes())

def __iter__(self):
    for key in self.list_attributes():
        yield key

def items(self):
    "Returns a list of all key and value pairs"
    return [(key, self[key]) for key in self]

def values(self):
    "Returns a list of all values"
    return [self[key] for key in self]

def keys(self):
    "Returns a list of all values"
    return self.list_attributes()

def to_dict(self):
    "Return a dict representation (copy) of all data"
    return dict(t for t in self.items())
%}
}

%extend dolfin::HDF5File {
%pythoncode %{
def __enter__(self) :
    return self

def __exit__(self, type, value, traceback) :
    self.close()
%}
}

%extend dolfin::X3DOMParameters
{
%pythoncode %{
def set_color_map(self, colormap):
    if (isinstance(colormap, str)):
        # If we are given a string, try to load the corresponding matplotlib cmap
        try:
            import matplotlib.cm
            import numpy
            mpl_cmap = matplotlib.cm.get_cmap(colormap)
            # Flatten colormap to simple list
            cmap_data = [val for s in [list(mpl_cmap(i)[:3]) for i in range(256)] for val in s]
            self._set_color_map(numpy.array(cmap_data, dtype='double'))
        except:
            # FIXME: raise error or print warning
            pass
    else:
        # Not a string - assume user has supplied valid cmap data as an array
        self._set_color_map(colormap)
%}
}


#endif

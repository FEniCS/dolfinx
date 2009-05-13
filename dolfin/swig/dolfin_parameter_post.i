%extend dolfin::NewParameters
{
 void _parse(PyObject *op)
 {
   if (PyList_Check(op)) {
     int i;
     int argc = PyList_Size(op);
     char **argv = (char **) malloc((argc+1)*sizeof(char *));
     for (i = 0; i < argc; i++) {
       PyObject *o = PyList_GetItem(op,i);
       if (PyString_Check(o))
	 argv[i] = PyString_AsString(o);
       else
       {
	 free(argv);
	 throw std::runtime_error("list must contain strings");
       }
     }
     argv[i] = 0;
     self->parse(argc, argv);
   } 
   else 
     throw std::runtime_error("not a list");
 }
    
%pythoncode%{

def parse(self,argv=None):
    "Parse command line arguments"
    if argv is None:
        import sys
        argv = sys.argv
    self._parse(argv)

def keys(self):
    "Returns a list of the parameter keys"
    ret = []
    _keys = STLVectorString()
    self._parameter_keys(_keys)
    for i in xrange(len(_keys)):
        ret.append(_keys[i])
    _keys.clear()
    self._database_keys(_keys)
    for i in xrange(len(_keys)):
        ret.append(_keys[i])
    return ret

def iterkeys(self):
    "Returns an iterator for the parameter keys"
    for key in self.keys():
        yield key

def values(self):
    "Returns a list of the parameter values"
    ret = []
    par_keys = STLVectorString()
    self._parameter_keys(par_keys)
    for key in self.keys():
        if key in par_keys:
            ret.append(self._get_parameter(key))
        else:
            ret.append(self._get_database(key))
            
def itervalues(self):
    "Returns an iterator to the parameter values"
    for val in self.values():
        yield val

def items(self):
    return zip(self.keys(),self.values())

def iteritems(self):
    for key, value in self.items():
        yield key, value

def set_range(self,key,*arg):
    "Set the range for the given parameter" 
    _keys = STLVectorString()
    self._parameter_keys(_keys)
    if key not in _keys:
        raise KeyError, "no parameter with name '%s'"%key
    self._get_parameter(key).set_range(*arg)

def __getitem__(self,key):
    "Return the parameter corresponding to the given key"
    _keys = STLVectorString()
    self._parameter_keys(_keys)
    if key in _keys:
        par = self._get_parameter(key)
        val_type = par.type_str()
        if val_type == "string":
            return str(par)
        elif val_type == "double":
            return float(par)
        elif  val_type == "int":
            return int(par)
        else:
            raise TypeError, "unknown value type '%s' of parameter '%s'"%(val_type,key)
    
    _keys.clear()
    self._database_keys(_keys)
    if key in _keys:
        return self._get_database(key)
    raise KeyError, "'%s'"%key


def __setitem__(self,key,value):
    "Set the parameter 'key', with given 'value'"
    _keys = STLVectorString()
    self._parameter_keys(_keys)
    if key not in _keys:
        raise KeyError, "%s is not a parameter"%key
    if not isinstance(value,(int,str,float)):
        raise TypeError, "can ony set 'int', 'float' and 'str' parameters"
    par = self._get_parameter(key)
    par._assign(value)

%}  
}

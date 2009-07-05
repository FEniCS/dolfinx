%pythoncode
%{

def debug(message):
    import traceback
    file, line, func, txt = traceback.extract_stack(None, 2)[0]
    __debug(file, line, func, message)

%}

%extend dolfin::Progress {

void __add(int incr) {
    for (int j=0;j<incr; ++j)
        (*self)++;
}

void __set(double value) {
    *self = value;
}

%pythoncode
%{
def __iadd__(self, other):
    if isinstance(other, int):
        self.__add(other)
    elif isinstance(other, float):
        self.__set(other)
    return self

def update(self, other):
    if isinstance(other, float):
        self.__set(other)
%}

}

%pythoncode %{
def info(*args):
    if len(args) > 0 and isinstance(args[0],(Variable,Parameters)):
        _info(args[0].__str__())
    else:
        _info(*args)
%}

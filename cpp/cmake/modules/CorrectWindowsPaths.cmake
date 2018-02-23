# CorrectWindowsPaths - this module defines one macro
#
# CONVERT_CYGWIN_PATH( PATH )
#  This uses the command cygpath (provided by cygwin) to convert
#  unix-style paths into paths useable by cmake on windows

macro (CONVERT_CYGWIN_PATH _path)
  if (WIN32)
    EXECUTE_PROCESS(COMMAND cygpath.exe -m ${${_path}}
      OUTPUT_VARIABLE ${_path})
    string (STRIP ${${_path}} ${_path})
  endif (WIN32)
endmacro (CONVERT_CYGWIN_PATH)


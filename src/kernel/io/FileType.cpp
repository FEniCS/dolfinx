// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "FileType.h"
#include "utils.h"

//-----------------------------------------------------------------------------
FileType GetFileType(const char *filename)
{
  FileType filetype;
  
  if ( suffix(filename,".inp") )
	 filetype = FILE_INP;
  else if ( suffix(filename,".dx") )
	 filetype = FILE_OPENDX;
  else if ( suffix(filename,".m") )
	 filetype = FILE_MATLAB;
  else if ( suffix(filename,".gid") )
	 filetype = FILE_GID;
  else
	 filetype = FILE_UNKNOWN;

  return filetype;
}
//-----------------------------------------------------------------------------

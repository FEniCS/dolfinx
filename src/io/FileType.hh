// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __FILE_TYPE_HH
#define __FILE_TYPE_HH

enum FileType { FILE_UNKNOWN, FILE_INP, FILE_OPENDX, FILE_MATLAB, FILE_GID };

FileType GetFileType(const char *filename);

#endif

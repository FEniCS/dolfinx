// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

// Contains small nonspecific utility functions useful for various
// tasks, such as string manipulation, simple type definitions, ...

#ifndef __UTILS_H
#define __UTILS_H

#include "kw_constants.h"
#include <stdio.h>

/// String manipulation
bool suffix(const char *string, const char *suffix);
void remove_newline(char *string);

/// File reading
bool end_of_file     (FILE *fp);
bool keyword_in_line (FILE *fp, const char *keyword);
bool keyword_in_line (FILE *fp, const char *keyword, char *line);
void skip_line       (FILE *fp);

/// Mathematics
int  factorial(int n);
real sqr(real x);
real max(real x, real y);
real min(real x, real y);

/// Other
bool contains(int *list, int size, int number);

/// System calls
void env_get_user(char *string);
void env_get_time(char *string);
void env_get_host(char *string);
void env_get_mach(char *string);
void env_get_name(char *string);
void env_get_vers(char *string);

#endif

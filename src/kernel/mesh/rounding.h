/* GTS - Library for the manipulation of triangulated surfaces
 * Copyright (C) 1999 Stéphane Popinet
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

//#include "config.h"

#ifdef HAVE_FPU_CONTROL_H
#  include <fpu_control.h>
#  ifdef _FPU_EXTENDED
#   if !defined(__alpha__) || !defined(__GLIBC__)
#    if defined(__arm__)
     static fpu_control_t fpu_round_double = _FPU_DEFAULT;
#    else
     static fpu_control_t fpu_round_double =
       (_FPU_DEFAULT & ~ _FPU_EXTENDED)|_FPU_DOUBLE;
#    endif
     static fpu_control_t fpu_init;
#    define FPU_ROUND_DOUBLE  { _FPU_GETCW(fpu_init);\
                                _FPU_SETCW(fpu_round_double); }
#    define FPU_RESTORE       {_FPU_SETCW(fpu_init);}
#   else /* __alpha__ && __GLIBC__ */
#    define FPU_ROUND_DOUBLE
#    define FPU_RESTORE
#   endif /* __alpha__ && __GLIBC__ */
#  else /* not FPU_EXTENDED */
#    define FPU_ROUND_DOUBLE
#    define FPU_RESTORE
#  endif /* not FPU_EXTENDED */
#else /* not HAVE_FPU_CONTROL_H */
#  ifdef __FreeBSD__
#    include <floatingpoint.h>
#    define FPU_ROUND_DOUBLE  (fpsetprec(FP_PD))
#    define FPU_RESTORE       (fpsetprec(FP_PE))
#  else /* not __FreeBSD__ */
#    ifdef WIN32
#      ifdef _MSC_VER
#        include <float.h>
         static unsigned int fpu_init;
#        define FPU_ROUND_DOUBLE (fpu_init = _controlfp (0, 0),\
                                 _controlfp (_PC_53, MCW_PC))
#        define FPU_RESTORE      (_controlfp (fpu_init, 0xfffff))
#      elif __MINGW32__
#        include <float.h>
         static unsigned int fpu_init;
#        define FPU_ROUND_DOUBLE (fpu_init = _controlfp (0, 0),\
                                  _controlfp (_PC_53, _MCW_PC))
#        define FPU_RESTORE      (_controlfp (fpu_init, 0xfffff))
#      else /* not _MSC_VER or __MINGW32__ */
#        error "You need MSVC or MinGW for the Win32 version"
#      endif /*  not _MSC_VER or __MINGW32__ */
#    else /* not WIN32 */
#      ifdef __CYGWIN__
         typedef unsigned int fpu_control_t __attribute__ ((__mode__ (__HI__)));
         static fpu_control_t fpu_round_double = 0x027f;
         static fpu_control_t fpu_init;
#        define _FPU_GETCW(cw) __asm__ ("fnstcw %0" : "=m" (*&cw))
#        define _FPU_SETCW(cw) __asm__ ("fldcw %0" : : "m" (*&cw))
#        define FPU_ROUND_DOUBLE  { _FPU_GETCW(fpu_init);\
                                    _FPU_SETCW(fpu_round_double); }
#        define FPU_RESTORE       { _FPU_SETCW(fpu_init);}
#      else /* not __CYGWIN__ */
#        ifdef CPP_HAS_WARNING
#          warning "Unknown CPU: assuming default double precision rounding"
#        endif /* CPP_HAS_WARNING */
#        define FPU_ROUND_DOUBLE
#        define FPU_RESTORE
#      endif /* not __CYGWIN__ */
#    endif /* not WIN32 */
#  endif /* not __FreeBSD__ */
#endif /* not HAVE_FPU_CONTROL_H */

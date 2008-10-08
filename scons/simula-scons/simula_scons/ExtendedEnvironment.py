# Copyright (c) 2006, Richard Levitte <richard@levitte.org>
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Modified by Johannes Ring 2008

import SCons
import re, os

class ExtendedEnvironment(SCons.Environment.Environment):
    #def __init__(self, *args, **kw):
    # Since SCons doesn't support versioning, we need to do it for SCons.
    # Unfortunately, that means having to do things differently depending
    # on platform...
    def VersionedSharedLibrary(self, libname, libversion, lib_objs=[]):
        platform = self.subst('$PLATFORM')
        shlib_pre_action = None
        shlib_suffix = self.subst('$SHLIBSUFFIX')
        shlib_prefix = self.subst('$SHLIBPREFIX')
        shlib_post_action = None
        shlib_post_action2 = None
        shlink_flags = SCons.Util.CLVar(self.subst('$SHLINKFLAGS'))
        major = libversion.split('.')[0]

        if platform == 'posix':
            shlib_post_action = ['rm -f $TARGET',
                                 'ln -s ${SOURCE.file} $TARGET']
            shlib_post_action2 = shlib_post_action
            shlib_post_action_output_re = \
                ['%s\\.[0-9\\.]*$' % re.escape(shlib_suffix), shlib_suffix]
            shlib_post_action_output_re2 = \
                ['%s\\.[0-9\\.]*$' % re.escape(shlib_suffix),
                 shlib_suffix + '.' + major]
            shlib_soname = shlib_prefix + os.path.basename(libname) \
                           + shlib_suffix + '.' + major
            shlib_suffix += '.' + libversion
            #shlink_flags += ['-Wl,-Bsymbolic', '-Wl,-soname=${TARGET.file}']
            shlink_flags += ['-Wl,-soname=' + shlib_soname]
        elif platform == 'aix':
            shlib_pre_action = [
                "nm -Pg $SOURCES > ${TARGET}.tmp1",
                "grep ' [BDT] ' < ${TARGET}.tmp1 > ${TARGET}.tmp2",
                "cut -f1 -d' ' < ${TARGET}.tmp2 > ${TARGET}",
                "rm -f ${TARGET}.tmp[12]" ]
            shlib_pre_action_output_re = [ '$', '.exp' ]
            shlib_post_action = [ 'rm -f $TARGET', 'ln -s $SOURCE $TARGET' ]
            shlib_post_action_output_re = [
                '%s\\.[0-9\\.]*' % re.escape(shlib_suffix),
                shlib_suffix ]
            shlib_suffix += '.' + libversion
            shlink_flags += ['-G', '-bE:${TARGET}.exp', '-bM:SRE']
        elif platform == 'cygwin':
            shlink_flags += [ '-Wl,-Bsymbolic',
                              '-Wl,--out-implib,${TARGET.base}.a' ]
        elif platform == 'darwin':
            #shlib_suffix = '.' + libversion + shlib_suffix
            #shlink_flags += [ '-dynamiclib',
            #                  '-current-version %s' % libversion ]
            pass

        lib = self.SharedLibrary(libname, lib_objs,
                                 SHLIBSUFFIX=shlib_suffix,
                                 SHLINKFLAGS=shlink_flags)

        if shlib_pre_action:
            shlib_pre_action_output = re.sub(shlib_pre_action_output_re[0],
                                             shlib_pre_action_output_re[1],
                                             str(lib[0]))
            self.Command(shlib_pre_action_output, [lib_objs], shlib_pre_action)
            self.Depends(lib, shlib_pre_action_output)
        if shlib_post_action:
            shlib_post_action_output = re.sub(shlib_post_action_output_re[0],
                                              shlib_post_action_output_re[1],
                                              str(lib[0]))
            sl = self.Command(shlib_post_action_output, lib, shlib_post_action)
            if shlib_post_action2:
                shlib_post_action_output2 = \
                    re.sub(shlib_post_action_output_re2[0],
                           shlib_post_action_output_re2[1], str(lib[0]))
                self.Command(shlib_post_action_output2, lib, shlib_post_action2)
                self.Depends(sl, shlib_post_action_output2)
        return lib

    def InstallVersionedSharedLibrary(self, destination, lib):
        platform = self.subst('$PLATFORM')
        shlib_suffix = self.subst('$SHLIBSUFFIX')
        shlib_install_pre_action = None
        shlib_install_post_action = None
        shlib_install_post_action2 = None

        if platform == 'posix':
            major = str(lib).split('.so.')[1].split('.')[0]
            shlib_post_action = ['rm -f $TARGET',
                                 'ln -s ${SOURCE.file} $TARGET']
            shlib_post_action2 = shlib_post_action
            shlib_post_action_output_re = \
                ['%s\\.[0-9\\.]*$' % re.escape(shlib_suffix), shlib_suffix]
            shlib_post_action_output_re2 = \
                ['%s\\.[0-9\\.]*$' % re.escape(shlib_suffix),
                 shlib_suffix + '.' + major]
            shlib_install_post_action = shlib_post_action
            shlib_install_post_action2 = shlib_post_action2
            shlib_install_post_action_output_re = shlib_post_action_output_re
            shlib_install_post_action_output_re2 = shlib_post_action_output_re2

        ilib = self.Install(destination, lib)

        if shlib_install_pre_action:
            shlib_install_pre_action_output = \
                re.sub(shlib_install_pre_action_output_re[0],
                       shlib_install_pre_action_output_re[1], str(ilib[0]))
            self.Command(shlib_install_pre_action_output, ilib,
                         shlib_install_pre_action)
            self.Depends(shlib_install_pre_action_output, ilib)
        if shlib_install_post_action:
            shlib_install_post_action_output = \
                re.sub(shlib_install_post_action_output_re[0],
                       shlib_install_post_action_output_re[1], str(ilib[0]))
            self.Command(shlib_install_post_action_output, ilib,
                         shlib_install_post_action)
            if shlib_install_post_action2:
                shlib_install_post_action_output2 = \
                    re.sub(shlib_install_post_action_output_re2[0],
                           shlib_install_post_action_output_re2[1],
                           str(ilib[0]))
                self.Command(shlib_install_post_action_output2, ilib,
                             shlib_install_post_action2)
            

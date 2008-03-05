import os.path, shutil
from SCons import Builder, Action

# _LatexAction = Action.Action("$LATEX -halt-on-error ${SOURCE.file} >/dev/null")
_LatexAction = Action.Action("$LATEX ${SOURCE.file}")
_ExtraSuffixes = ("aux", "log", "out", "toc")
_DviPdfAction = Action.CommandAction("$DVIPDFCOM")
_DviPsAction = Action.CommandAction("$DVIPSCOM")

def _genDvi(target, source, env):
    """ Generate DVI from source LaTeX file.
    @param source: Source nodes.
    @param env: SCons Environment.
    """
    srcName = os.path.splitext(os.path.basename(str(source[0])))[0]
    extraFnames = ["%s.%s" % (srcName, ext) for ext in _ExtraSuffixes]
    fnames = ["%s.%s" % (srcName, "dvi")] + extraFnames

    i = 0
    maxPasses = 3
    while i < maxPasses:
        try:
            assert os.path.exists(os.path.basename(str(source[0])))
            ret = _LatexAction([], source, env)
            if ret:
                return ret
        except Exception, err:
            for n in fnames:
                try: os.remove(n)
                except Exception: pass
            raise

        # Check if we need another pass
        found = False
        f = open("%s.log" % srcName)
        try:
            for l in f:
                if l.strip() == "No file %s.aux." % srcName:
                    found = True
                    break
        finally:
            f.close()
        if not found:
            # Good
            break

        i += 1

    if i >= maxPasses:
        raise Exception, "%d passes was not enough" % i

def _actionFunc(target, source, env, converter):
    tgt = str(target[0])

    # Copy source files to build directory as necessary

    copied = []
    tgtDir = os.path.dirname(tgt)
    try:
        for s in [str(s) for s in source]:
            if tgtDir != os.path.dirname(s):
                # Copy source to destination directory
                dest = os.path.join(tgtDir, os.path.basename(s))
                shutil.copy(s, dest)
                copied.append(dest)

        # Generate document in build directory

        oldDir = os.getcwd()
        os.chdir(tgtDir)
        try:
            result = converter(target, source, env)
        finally:
            os.chdir(oldDir)
    finally:
        # Remove copied source files
        for fname in copied:
            try: os.remove(fname)
            except OSError: pass

    return result

def latexFunc(target, source, env):
    """ Build document from LaTeX sources. """
    return _actionFunc(target, source, env, _genDvi)

def emitter(target, source, env):
    """ Emit additional targets for LaTeX source. """
    srcName = os.path.splitext(str(source[0]))[0]
    extraTargets = ["%s.%s" % (srcName, ext) for ext in _ExtraSuffixes]
    return (target + extraTargets, source)

def _dviPdfFunc(target, source, env):
    """ Convert DVI into PDF.
    
    We define this function ourselves, since the SCons implementation invokes dvipdf from the
    top-level directory which only works if the source is in the same directory.
    """
    return _actionFunc(target, source, env, _DviPdfAction)

def _dviPsFunc(target, source, env):
    """ Convert DVI into Postscript.
    
    We define this function ourselves, since the SCons implementation invokes dvips from the
    top-level directory which only works if the source is in the same directory.
    """
    return _actionFunc(target, source, env, _DviPsAction)

def _dviEmitter(target, source, env):
    """ Strip off any other sources than .dvi (.aux, .log etc.). """
    source = [s for s in source if os.path.splitext(str(s))[1] == ".dvi"]
    return (target, source)

def _stringFunc(target, source, env):
    return "Building %s from %s" % (target[0], ", ".join([str(s) for s in source]))

def generate(env):
    bld = env["BUILDERS"]["DVI"] = Builder.Builder(action={}, emitter={}, suffix=".dvi")
    for sufx in (".latex", ".ltx", ".tex"):
        bld.add_action(sufx, Action.Action(latexFunc, strfunction=_stringFunc))
        bld.add_emitter(sufx, emitter)

    # Note that when setting DVIPDFCOM and DVIPSCOM we use unqualified filenames since we
    # change to the target directory before building.

    bld = env['BUILDERS']['PDF'] = Builder.Builder(action={}, emitter={}, prefix='$PDFPREFIX', suffix='$PDFSUFFIX')
    bld.add_emitter(".dvi", _dviEmitter)
    bld.add_action(".dvi", Action.Action(_dviPdfFunc, strfunction=_stringFunc))
    env['DVIPDFCOM'] = '$DVIPDF $DVIPDFFLAGS ${SOURCE.file} ${TARGET.file}'

    # From dvips.py, looks like we have to construct our own Builder
    bld = Builder.Builder(action=Action.Action(_dviPsFunc, strfunction=_stringFunc), prefix='$PSPREFIX', suffix='$PSSUFFIX', \
            src_suffix='.dvi', src_builder='DVI', emitter=_dviEmitter)
    env["BUILDERS"]["PostScript"] = bld
    env['DVIPSCOM'] = '$DVIPS $DVIPSFLAGS -o ${TARGET.file} ${SOURCE.file}'

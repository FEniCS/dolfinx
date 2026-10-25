# DOLFINx C++ interface documentation

The C++ API documentation is generated from the source code using
[Doxygen](https://www.doxygen.nl/), and the Doxygen output is curated and
rendered using reStructured text with [Sphinx](https://www.sphinx-doc.org/) and
[Breathe](https://breathe.readthedocs.io/).


## Requirements

For the basic C++ documentation:

- [Doxygen](https://www.doxygen.nl/)

And then for the nicer-looking Sphinx documentation, Python with:

- [sphinx](https://www.sphinx-doc.org/)
- [breathe](https://breathe.readthedocs.io/)
- [pydata-sphinx-theme](https://pydata-sphinx-theme.readthedocs.io/en/stable/)
- [myst-parser](https://myst-parser.readthedocs.io/en/latest/)

## Building

```bash
doxygen Doxyfile
python -m sphinx -b html source/ build/html/
```

Sphinx/Breathe documentation is in `build/html` and the Doxygen generated
HTML is in `html/`.

## Checking

Sphinx does not parse LaTeX; MathJax typesets it in the browser. To
check that MathJax can typeset all math in the built documentation
(requires [Node.js](https://nodejs.org/)):

```bash
npm install --no-save @mathjax/src@4.1.3
node check_math.mjs build/html
```

Converting the demos fails if a code comment contains MyST markup such
as `$...$` or `{cpp:func}`, which is rendered only in Markdown cells.

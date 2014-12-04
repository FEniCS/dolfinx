#!/bin/sh

# Clean up frames in movie
rm -rf tmp_*.png  

# Should have kappa_0=2.3 or 12.3 and kappa_1=kappa_0 or 1E+4
# (physical values for the rest)
python sin_daD.py   1   1.5  4  40

# Make movie as an HTML file
scitools movie output_file=movie.html fps=2 tmp_*.png
echo "Run google-chrome movie.html"

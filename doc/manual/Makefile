FILENAME='dolfin-user-manual'

all:
	latex $(FILENAME).tex
	makeindex $(FILENAME)
	bibtex $(FILENAME)
	latex $(FILENAME)
	dvips -P pdf -o $(FILENAME).ps $(FILENAME).dvi
	ps2pdf $(FILENAME).ps $(FILENAME).pdf

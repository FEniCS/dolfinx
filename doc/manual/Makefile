FILENAME='dolfin-user-manual'

all:
	latex $(FILENAME).tex
	bibtex $(FILENAME)
	makeindex $(FILENAME)
	latex $(FILENAME)
#	dvips -P pdf -o $(FILENAME).ps $(FILENAME).dvi
	dvips -o $(FILENAME).ps $(FILENAME).dvi
	ps2pdf $(FILENAME).ps $(FILENAME).pdf

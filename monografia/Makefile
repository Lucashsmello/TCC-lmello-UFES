DOC:=monografia
TMPDIR:=.tmpdir

all: $(TMPDIR) $(DOC).tex
	pdflatex -output-directory=$(TMPDIR) $(DOC).tex
	bibtex $(TMPDIR)/$(DOC)
	bibtex $(TMPDIR)/$(DOC)
	pdflatex -output-directory=$(TMPDIR) $(DOC).tex
	bibtex $(TMPDIR)/$(DOC)
	pdflatex -output-directory=$(TMPDIR) $(DOC).tex
	mv $(TMPDIR)/$(DOC).pdf .
#	dvips $(TMPDIR)/$(DOC).dvi -o $(TMPDIR)/$(DOC).ps
#	ps2pdf $(TMPDIR)/$(DOC).ps


$(TMPDIR):
	mkdir .tmpdir

clean:
	rm -rf $(TMPDIR)

#latex
#latex
#dvips
#ps2pdf

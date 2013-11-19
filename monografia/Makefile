DOC:=monografia
TMPDIR:=.tmpdir

all: $(TMPDIR) $(DOC).tex
	latex -output-directory=$(TMPDIR) $(DOC).tex
	latex -output-directory=$(TMPDIR) $(DOC).tex
	dvips $(TMPDIR)/$(DOC).dvi -o $(TMPDIR)/$(DOC).ps
	ps2pdf $(TMPDIR)/$(DOC).ps


$(TMPDIR):
	mkdir .tmpdir

clean:
	rm -rf $(TMPDIR)

#latex
#latex
#dvips
#ps2pdf

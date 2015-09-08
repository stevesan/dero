
TESTOUT=testing
testwad:
	mkdir $(TESTOUT) || echo blah
	cd $(TESTOUT) && python ../wad.py && ../run_wad.sh square-built.wad

WORKDIR=WORK
gen:
	mkdir $(WORKDIR) || echo already exists, ok
	rm $(WORKDIR)/locks*.png || echo ok
	cd $(WORKDIR) && python ../gen.py 64 8 && ../run_wad.sh built-playable.wad

SHAPEDIR=SHAPEGEN
shape:
	mkdir $(SHAPEDIR) || echo already exists, ok
	rm $(SHAPEDIR)/*.png || echo ok
	cd $(SHAPEDIR) && python ../shape-gen.py

drawwads:
	cd allmaps && python ../wad.py


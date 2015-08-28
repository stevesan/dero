
TESTOUT=testing
testwad:
	mkdir $(TESTOUT) || echo blah
	cd $(TESTOUT) && python ../wad.py && ../run_wad.sh square-built.wad

WORKDIR=WORK
gen:
	mkdir $(WORKDIR) || echo already exists, ok
	rm $(WORKDIR)/locks*.png || echo ok
	cd $(WORKDIR) && python ../gen.py 100 10 && ../run_wad.sh built-playable.wad

drawwads:
	cd allmaps && python ../wad.py


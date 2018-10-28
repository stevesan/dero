
TESTOUT=testing
testwad:
	mkdir $(TESTOUT) || echo blah
	cd $(TESTOUT) && python ../wad.py && ../run_wad.sh square-built.wad

WORKDIR=WORK
gen:
	mkdir $(WORKDIR) || echo already exists, ok
	rm $(WORKDIR)/locks*.png || echo ok
	cd $(WORKDIR) && python ../gen.py 64 8 && ../run_wad.sh built-playable.wad

SHAPEDIR=SHAPE_GEN_OUT
shapegen:
	mkdir $(SHAPEDIR) || echo already exists, ok
	rm $(SHAPEDIR)/*.png || echo ok
	cd $(SHAPEDIR) && python ../shapegen.py

DUMPDIR=DUMP_WADS_OUT
dumpwads:
	mkdir -p $(DUMPDIR)
	cd $(DUMPDIR) && python ../wad.py

csg:
	mkdir -p CSG_OUT
	cd CSG_OUT && python ../csg.py

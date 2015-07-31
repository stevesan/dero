
TESTOUT=testing
testwad:
	mkdir $(TESTOUT) || echo blah
	cd $(TESTOUT) && python ../wad.py && ../run_wad.sh square-built.wad

GENOUT=genout
gen:
	rm -rf $(GENOUT) || echo blah
	mkdir $(GENOUT)
	cd $(GENOUT) && python ../gen.py 100 10 && ../run_wad.sh gen-built.wad

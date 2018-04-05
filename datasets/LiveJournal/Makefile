WGET = wget
UNZIP = gzip -d
GRAPH_ZIP_FILE = soc-LiveJournal1.txt.gz
GRAPH_URL = http://snap.stanford.edu/data/soc-LiveJournal1.txt.gz

all: setup

setup: download
	$(UNZIP) $(GRAPH_ZIP_FILE)
	
download:
	$(WGET) $(GRAPH_URL)

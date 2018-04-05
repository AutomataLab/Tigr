WGET = wget
UNZIP = gzip -d
GRAPH_ZIP_FILE = soc-pokec-relationships.txt.gz
GRAPH_URL = http://snap.stanford.edu/data/soc-pokec-relationships.txt.gz

all: setup

setup: download
	$(UNZIP) $(GRAPH_ZIP_FILE)
	
download:
	$(WGET) $(GRAPH_URL)

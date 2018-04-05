WGET = wget
UNZIP = gzip -d
GRAPH_ZIP_FILE = higgs-social_network.edgelist.gz
GRAPH_URL = http://snap.stanford.edu/data/higgs-social_network.edgelist.gz

all: setup

setup: download
	$(UNZIP) $(GRAPH_ZIP_FILE)
	
download:
	$(WGET) $(GRAPH_URL)

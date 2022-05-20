
docker :
	docker build -t d4gen .
	docker run -it d4gen /bin/bash

eval :
	python3 cli.py eval $(MODEL) $(ARGS)

load_from_vm :
	rsync -ovapx cloudadm@vm3-d4gen.vm.fedcloud.eu:/opt/cloudadm/DL-CancerLncRNA/weights/ ./weights/

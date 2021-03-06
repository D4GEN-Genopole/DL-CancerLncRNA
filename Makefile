
docker :
	docker build -t d4gen .
	docker run -it -p 8051:8051 d4gen /bin/bash

eval :
	python3 cli.py eval $(MODEL) $(ARGS)

load_from_vm :
	rsync -ovapx cloudadm@vm3-d4gen.vm.fedcloud.eu:/opt/cloudadm/DL-CancerLncRNA/weights/ ./weights/

deploy_api :
	# Make test prediction
	streamlit run api/app.py

viz :
	python3 cli.py visualize_data
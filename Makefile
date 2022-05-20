
docker :
	docker build -t d4gen .
	docker run -it d4gen /bin/bash

eval :
	python3 cli.py eval $(MODEL) $(ARGS)
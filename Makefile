
docker :
	docker build -t d4gen .
	docker run -it d4gen /bin/bash

train_seq :
	python3 main_sequences.py

train_exp :
	python3 main_expressions.py
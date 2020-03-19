SHELL		:= /bin/bash


train-local:
	python vpg.py

train-cloud:
	az ml run submit-script -c train -e vpg_cartpole --source-directory .

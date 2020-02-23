SHELL		:= /bin/bash


eval-local:
	python watch_agent_play.py

eval-cloud:
	az ml run submit-script -c eval -e dqn_atari --source-directory .

train-local:
	python dqn.py

train-cloud:
	az ml run submit-script -c test -e dqn_atari --source-directory .

SHELL		:= /bin/bash


dqn-local-eval:
	python watch_agent_play.py

dqn-cloud-eval:
	az ml run submit-script -c eval -e dqn_atari --source-directory .

dqn-local-train:
	python dqn.py

dqn-cloud-train:
	az ml run submit-script -c test -e dqn_atari --source-directory .

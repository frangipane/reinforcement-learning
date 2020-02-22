"""Script to generate a runconfig for the Azure ML CLI.

1. First attach the directory(workspace) to Azure.
2. Then run this script in the attached workspace.

This script assumes many things, e.g.

1. assumes the directory contains environment.yml, a conda environment
file, and a python script called dqn.py.  

2. Assumes you have an existing compute target called "gpu-cluster".
Alternatively, you can specify "amlcompute" with vm_size.  The vm_size in this 
example does not actually do anything because it's using an existing target.

https://github.com/microsoft/MLOps/blob/master/examples/cli-train-deploy/generate-runconfig.py

To use the output of the script

1. cd to the workspace directory
2. Make sure you don't have your conda environment activated
3. Type on the command line:
az ml run submit-script -c test -e dqn_atari --source-directory .

Note, the input for -c is "test", which corresponds to "test.runconfig" generated
by this script.
"""
import os

from azureml.core import Workspace, Run, Environment, ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.experiment import Experiment
from azureml.core.runconfig import RunConfiguration
from azureml.core.script_run_config import get_run_config_from_script_run

# load Workspace
ws = Workspace.from_config()

# load Experiment
experiment = Experiment(workspace=ws, name='test-expt')

# Create python environment for Azure machine learning expt
# options for class methods: from_conda_specification, from_pip_requirements,
# from_existing_conda_environment
myenv = Environment(name="test")
# myenv = Environment.from_conda_specification(name="test",
#                                              file_path="./environment.yml")

# Environment: docker section
# docker_config = dict(
#     enabled=True,
#     base_image="base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04",
#     # comment out this environment variable if you don't have it set!
#     environment_variables={'WANDB_API_KEY': os.environ['WANDB_API_KEY']}
# )
# docker_section = DockerSection(**docker_config)


## Environment: docker section
myenv.docker.enabled = True
myenv.docker.base_image = "mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04"
# comment out this environment variable if you don't have it set!
myenv.environment_variables = {'WANDB_API_KEY': os.environ['WANDB_API_KEY']}

## Environment: python section
conda_dep = CondaDependencies(conda_dependencies_file_path='environment.yml')
myenv.python.conda_dependencies = conda_dep
# myenv.python.conda_dependencies_file = 'environment.yml'

# create configuration for Run
# Use RunConfiguration to specify compute target / env deps part of run
run_config = RunConfiguration()

# Attach compute target to run config
run_config.framework = 'python'
run_config.target = "gpu-cluster"
# This doesn't actuallly do anything since my target is a persistent compute instead of amlcompute
run_config.amlcompute.vm_size = "Standard_NC24"
run_config.node_count = 1
run_config.environment = myenv


# ScriptRunConfig packaages together environment configuration of
# RunConfiguration with a script for training to create a **script** run.
""" RunConfiguration(script=None, arguments=None, framework=None, communicator=None,
 conda_dependencies=None, _history_enabled=None, _path=None, _name=None)
"""
src = ScriptRunConfig(source_directory='.',
                      script='dqn.py',
                      run_config=run_config)


get_run_config_from_script_run(src).save(path='./test.runconfig')

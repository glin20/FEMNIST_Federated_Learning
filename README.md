# Federated Learning using Flower

## 1. Setup a virtual environment and install packages
### Conda
1. Download the latest Conda version here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
2. Initialize conda through `conda init`
3. cd into cloned repository into directory containing requirements file
4. Run `conda create -n <name> --f requirements.yaml`
5. Activate the environment using `conda activate <env_name>`
### venv
1. Make sure the python version is >= 3.10.0
2. Create an environment using `python -m venv <env_name>`
3. cd into the directory into Scripts
4. Activate one of the "activate" files depending on your terminal
5. cd into cloned repository into directory containing requirements file
6. Run `pip install -r requirements.txt`

## 2. Run the Simulation
1. cd  into the cloned repository into /src
2. Change whether or not you want the simulation to run on CPU or GPU in src.py
3. Change whether or not you want data poisoning in client.py
4. run `python sim.py`
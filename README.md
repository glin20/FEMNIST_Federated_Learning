# Federated Learning using Flower

## (Optional) Setup a virtual environment using Conda
1. Download the latest Conda version here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
2. Initialize conda through `conda init`
3. Create an environment in your terminal using `conda create -n <env_name>`
4. Activate the environment using `conda activate <env_name>`

## 1. Install the necessary packages
### Conda
1.  `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
2. `conda install -q matplotlib` 
3. `pip install flwr_datasets[vision] flwr[simulation]"`
### Pip
1. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
2. `pip install -q flwr_datasets[vision] flwr[simulation] matplotlib`

## 2. Run the Simulation
1. cd  into the cloned repository
2. run `python sim.py`
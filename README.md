# Deep Reinforcement Learning for Smart Cities

![smart-cities-drl](assets/wastenet-routes.png)

## Documentation

RLlib: https://docs.ray.io/en/master/rllib.html

Mesa: https://mesa.readthedocs.io/en/stable/


## Installation

Clone repository and install dependencies.

```
git clone git@github.com:eescriba/smart-cities-drl.git
cd smart-cities-drl
python3 -m venv venv
source env/bin/activate
pip install -r requirements.txt
```

## Training

Train environments in Jupyter notebooks with RLlib.

* Local: `/notebooks/wastenet.ipynb`
* Colab: [![WasteNet](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eescriba/smart-cities-drl/blob/master/notebooks/wastenet.ipynb)


## Simulations

Run and visualize environments with Mesa.
```
mesa runserver src/wastenet
```
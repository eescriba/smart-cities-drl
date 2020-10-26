import sys
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import TaxiEnv
import numpy as np

from taxi_dqn.server import server


class MesaTaxiEnv(TaxiEnv):
    """
    Gym Taxi Environment with Mesa rendering.
    """

    def launch(self):
        server.launch()
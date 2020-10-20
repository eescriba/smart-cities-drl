import sys
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import TaxiEnv
import numpy as np

from taxi_ql.server import server


class MesaTaxiEnv(TaxiEnv):
    """
    Gym Taxi Environment with Mesa rendering.
    """

    def launch(self):
        server.launch()

        # outfile = StringIO() if mode == 'ansi' else sys.stdout

        # out = self.desc.copy().tolist()
        # out = [[c.decode('utf-8') for c in line] for line in out]
        # taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        # def ul(x): return "_" if x == " " else x
        # if pass_idx < 4:
        #     out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
        #         out[1 + taxi_row][2 * taxi_col + 1], 'yellow', highlight=True)
        #     pi, pj = self.locs[pass_idx]
        #     out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
        # else:  # passenger in taxi
        #     out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
        #         ul(out[1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)

        # di, dj = self.locs[dest_idx]
        # out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        # outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        # if self.lastaction is not None:
        #     outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        # else: outfile.write("\n")

        # # No need to return anything for human
        # if mode != 'human':
        #     with closing(outfile):
        #         return outfile.getvalue()
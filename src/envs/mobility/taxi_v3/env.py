from gym.envs.toy_text.taxi import TaxiEnv


class MesaTaxiEnv(TaxiEnv):
    def __init__(self):
        self.locs_colors = ["Y", "R", "B", "G"]
        super().__init__()

    @property
    def shape(self):
        width = int((len(self.desc[0]) - 1) / 2)
        height = len(self.desc) - 2
        return width, height

    @property
    def cells(self):
        return self.desc[1:-1, 1:-1].copy()


env = MesaTaxiEnv()

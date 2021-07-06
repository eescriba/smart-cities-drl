from mesa.space import MultiGrid


class SmartCabMultiGrid(MultiGrid):
    """MultiGrid with transformed coordinates"""

    def move_agent(self, agent, pos):
        return super().move_agent(agent, self.transform_grid_coords(pos))

    def place_agent(self, agent, pos):
        return super().place_agent(agent, self.transform_grid_coords(pos))

    def transform_grid_coords(self, pos):
        x, y = pos
        return (y, (self.width - x) % self.width - 1)

    def get_heading(self, current, pos, dest):
        dest = self.transform_grid_coords(dest)
        x, y = pos[0] - dest[0], pos[1] - dest[1]
        if x == 0:
            if y == 0:
                return current
            elif y > 0:
                return "S"
            else:
                return "N"
        elif x > 0:
            return "W"
        else:
            return "E"

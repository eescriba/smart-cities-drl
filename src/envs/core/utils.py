def get_heading(current, pos, dest):
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


def transform_coords(x, y):
    return (int(y / 2), (5 - x) % 5 - 1)

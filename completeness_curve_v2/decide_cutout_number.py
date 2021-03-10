def decide_cutout_number(z):
    z = float(z)
    if z < 0.5:
        n_cutout = 25
        cutout_size = 110.0
    elif z < 0.7:
        n_cutout = 35
        cutout_size = 80.0
    elif z < 0.9:
        n_cutout = 42
        cutout_size = 65.0
    else:
        n_cutout = 0
        cutout_size = 65.0

    return n_cutout, cutout_size

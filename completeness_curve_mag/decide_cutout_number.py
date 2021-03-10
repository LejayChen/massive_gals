def decide_cutout_number(z):
    z = float(z)
    if z<0.5:
        n_cutout = 40
        cutout_size = 110.0
    if z < 0.5:
        n_cutout = 50
        cutout_size = 80.0
    if z < 0.5:
        n_cutout = 60
        cutout_size = 65.0

    return n_cutout, cutout_size

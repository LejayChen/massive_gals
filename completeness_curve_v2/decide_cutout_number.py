def decide_cutout_number(z):
    z = float(z)
    if z<0.25:
        n_cutout = 30
    elif z<0.35:
        n_cutout = 25
    elif z<0.45:
        n_cutout = 22
    elif z<0.55:
        n_cutout = 20
    elif z<0.65:
        n_cutout = 20
    elif z<0.75:
        n_cutout = 18
    elif z<0.85:
        n_cutout = 15
    else:
        n_cutout = 10

    return n_cutout

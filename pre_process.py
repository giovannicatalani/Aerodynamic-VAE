# preprocess_zenodo_data.py

import numpy as np
from scipy.interpolate import griddata

def interpolate_to_grid(pressure, x_coords, y_coords, res=128):
    """
    Interpolate unstructured CFD data onto a regular pixel grid.
    x_coords, y_coords: (n_samples, n_nodes) - coordinates vary per sample
    pressure:           (n_samples, n_nodes)
    """
    x_lin = np.linspace(-0.25, 1.25, res)
    y_lin = np.linspace(-0.5, 0.5, res)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)

    n_samples = pressure.shape[0]
    pressure_grid = np.zeros((n_samples, res, res), dtype=np.float32)

    for i in range(n_samples):
        if i % 50 == 0:
            print(f"  Interpolating sample {i}/{n_samples}...")

        # Each sample has its own set of scattered points
        points = np.column_stack([x_coords[i], y_coords[i]])  # (n_nodes, 2)
        values = pressure[i]                                    # (n_nodes,)

        pressure_grid[i] = griddata(
            points, values, (X_grid, Y_grid),
            method='linear', fill_value=0.0
        )

    return pressure_grid


def convert_dataset(data_directory, res=128):
    for split in ['cyc', 'random']:
        src_file = data_directory + f'db_{split}.npy'
        dst_file = data_directory + f'db_{split}_{res}.npy'

        print(f"\nProcessing {src_file} ...")
        db = np.load(src_file, allow_pickle=True).item()

        pressure = db['Pressure']       # (n_samples, 27499)
        x_coords = db['Xcoordinate']    # (n_samples, 27499)
        y_coords = db['Ycoordinate']    # (n_samples, 27499)
        alpha    = db['Alpha']
        vinf     = db['Vinf']

        pressure_interp = interpolate_to_grid(pressure, x_coords, y_coords, res)

        db_out = {
            f'P_{res}x{res}': pressure_interp,
            'Alpha': alpha,
            'Vinf': vinf,
        }
        np.save(dst_file, db_out)
        print(f"  Saved {pressure_interp.shape} -> {dst_file}")


if __name__ == '__main__':
    data_directory = '../data/'
    convert_dataset(data_directory, res=128)
    print("\nDone!")
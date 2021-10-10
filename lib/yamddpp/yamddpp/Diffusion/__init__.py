import numpy as np
# from simpletraj.dcd.dcd import DCDReader  # or self-made dcd reader
from MDAnalysis.coordinates.DCD import DCDReader

from .mutual_diffusion import cu_mutual_diffusion
from .self_diffusion import mqd
from .self_diffusion import msd
from .self_diffusion import msd_square

_BUFFER_SIZE = 200000000  # 2e8 coordinates for n_frames * n_particles


def traj_dcd(dcd_file, func=msd, cum=True):
    r"""Example of dcd input.
    :param dcd_file: str, dcd_file
    :param func: callable, msd or mqd
    :param cum: bool
    :return: np.ndarray, msd
    """
    dcd = DCDReader(dcd_file)
    if dcd.periodic:
        raise ValueError("Error, periodic data found!")
    n_samples = dcd.numatoms
    n_frames = dcd.numframes
    _BUFFER = _BUFFER_SIZE // n_frames
    n_buffer = n_samples // _BUFFER
    _shape = (n_frames,) if cum else (n_frames, n_samples)
    ret = np.zeros(_shape)
    counter = 0
    for _ in range(n_buffer):
        x = np.asarray([np.copy(frame[counter:counter + _BUFFER]) for frame in dcd])
        x = np.asarray(x, dtype=np.float64)  # float64 is needed for accuracy, for simpletraj
        # gives np.float32 in reading hoomd dcd file.
        counter += _BUFFER
        ret += func(x, cum=cum)
    x = np.asarray([np.copy(_[counter:]) for _ in dcd])
    ret += func(x, cum=cum)
    return ret / n_samples

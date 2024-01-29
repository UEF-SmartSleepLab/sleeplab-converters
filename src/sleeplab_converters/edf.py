"""Tools for reading .edf data."""
import numpy as np
import pyedflib

from functools import partial
from pathlib import Path
from typing import Any, Optional


def read_signal_from_path(
        edf_path: str,
        idx: int,
        digital: bool = False,
        dtype: np.dtype = np.float32) -> np.array:
    with pyedflib.EdfReader(edf_path, annotations_mode=0) as hdl:
        # Read as digital if need to rewrite EDF
        # since otherwise will crash due to shifted values
        # https://github.com/holgern/pyedflib/issues/46
        s = hdl.readSignal(idx, digital=digital)

    return np.array(s).astype(dtype)


def read_signal_from_hdl(
        handle: pyedflib.EdfReader, idx: int, digital: bool = False) -> np.array:
    # Read as digital if need to rewrite EDF
    # since otherwise will crash due to shifted values
    # https://github.com/holgern/pyedflib/issues/46
    s = handle.readSignal(idx, digital=digital)

    return np.array(s)


def read_edf_export(edf_path: Path,
                    digital: bool = False,
                    ch_names: Optional[list[str]] = None,
                    annotations: bool = False,
                    dtype: np.dtype = np.float32
                    ) -> tuple[list[np.array], list[dict[str, Any]], dict[str, Any]]:
    """Read the EDF file and return signals and headers separately.
    
    Instead of the actual signal, return a function which reads the signal
    when evaluated. This way, all data need not fit into memory.
    """
    edf_path_str = str(edf_path.resolve())

    # Tell EdfReader not to validate annotations if they will not be used
    if annotations is False:
        annotations_mode = 0
    else:
        annotations_mode = 2

    with pyedflib.EdfReader(edf_path_str, annotations_mode=annotations_mode) as hdl:
        n_chs = hdl.signals_in_file
    
        # Resolve the channel indices if channel names are given
        if ch_names is None:
            # Defaults to all channels
            ch_idx = range(n_chs)
        else:
            # Create a mapping from channel name to channel index
            ch_name_idx_map = {}
            for i in range(n_chs):
                ch_name_idx_map[hdl.getLabel(i).strip()] = i
            ch_idx = [ch_name_idx_map[ch_name] for ch_name in ch_names]
            
        header = hdl.getHeader()

        # add annotations to header
        if annotations:
            annotations = hdl.readAnnotations()
            annotations = [[s, d, a] for s,d,a in zip(*annotations)]
            header['annotations'] = annotations

        signal_headers = []
        s_load_funcs = []
        for i in ch_idx:
            s_header = hdl.getSignalHeader(i)
            fs = hdl.samples_in_datarecord(i) / hdl.datarecord_duration
            # Patch the wrongly calculated fs
            s_header['sample_frequency'] = fs
            signal_headers.append(s_header)
            
            s_func = partial(
                read_signal_from_path, edf_path=edf_path_str, idx=i, digital=digital, dtype=dtype)
            # s_func = partial(
            #     read_signal_from_hdl, handle=hdl, idx=i, digital=digital)
            s_load_funcs.append(s_func)
            
    return s_load_funcs, signal_headers, header
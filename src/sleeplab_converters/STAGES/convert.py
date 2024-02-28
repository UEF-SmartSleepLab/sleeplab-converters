import argparse
import logging
import numpy as np
import pandas as pd

from datetime import datetime
from pathlib import Path
from sleeplab_converters import edf
from sleeplab_format import models, writer
from typing import Any, Callable


logger = logging.getLogger(__name__)


def parse_csv_BOGN():
    stage_map = {}
    event_map = {}

    # Interpret the 'cal' events as analysis_start and analysis_end


def parse_edf(edf_path: Path) -> tuple[datetime, ]:
    """Read the start_ts and SampleArrays from the EDF."""
    def _parse_samplearray(
            _load_func: Callable[[], np.array],
            _header: dict[str, Any]) -> models.SampleArray:
        array_attributes = models.ArrayAttributes(
            # Replace '/' and space with '_' to avoid errors in filepaths
            name=_header['label'].replace('/', '_').replace('\s', '_'),
            start_ts=start_ts,
            sampling_rate=_header['sample_frequency'],
            unit=_header['dimension']
        )
        return models.SampleArray(attributes=array_attributes, values_func=_load_func)

    s_load_funcs, s_headers, header = edf.read_edf_export(edf_path)

    start_ts = header['startdate']
    sample_arrays = {}
    for s_load_func, s_header in zip(s_load_funcs, s_headers):
        sample_array = _parse_samplearray(s_load_func, s_header)
        sample_arrays[sample_array.attributes.name] = sample_array

    return start_ts, sample_arrays


def convert_BOGN():
    subjects = {}
    # Loop through csv files
    for csvpath in ...:
        # If there is no edf corresponding to csv, skip
        if ...:
            continue

        start_ts, sample_arrays = parse_edf(...)
        
        sid, events, hypnogram, analysis_start, analysis_end = parse_csv_BOGN(...)

        subject = models.Subject(
            metadata=...,
            sample_arrays=...,
            annotations=...
        )

        subjects[...] = subject

    series = ...
    
    return series


def convert_GS():
    pass


def convert_MS():
    pass


def convert_STLK():
    pass


def convert_STNF():
    pass


CONVERSION_FUNCS = {
    'BOGN': convert_BOGN,
    'GSBB': convert_GS,
    'GSDV': convert_GS,
    'GSLH': convert_GS,
    'GSSA': convert_GS,
    'GSSW': convert_GS,
    'MSMI': convert_MS,
    'MSNF': convert_MS,
    'MSQW': convert_MS,
    'MSTH': convert_MS,
    'MSTR': convert_MS,
    'SLTK': convert_STLK,
    'STNF': convert_STNF
}


ALL_SERIES = list(CONVERSION_FUNCS.keys())


def convert_dataset(
        src_dir: Path,
        dst_dir: Path,
        ds_name: str = 'STAGES',
        series: list[str] = ALL_SERIES,
        array_format: str = 'zarr',
        clevel: int = 7,
        annotation_format: str = 'json') -> None:
    series_dict = {}
    for series_name in series:
        logger.info(f'Converting series {series_name}...')
        series_dict[series] = CONVERSION_FUNCS[series_name](
            ...
        )

    dataset = models.Dataset(name=ds_name, series=series)
    logger.info(f'Writing dataset {ds_name} to {dst_dir}')
    writer.write_dataset(
        dataset,
        basedir=dst_dir,
        annotation_format=annotation_format,
        array_format=array_format,
        compression_level=clevel
    )


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', type=Path, required=True,
        help='The root folder of the STAGES dataset.')
    parser.add_argument('--dst-dir', type=Path, required=True,
        help='The rood folder where the SLF dataset is saved.')
    parser.add_argument('--ds-name', default='STAGES',
        help='The name of the resulting SLF dataset.')
    parser.add_argument('--series', nargs='*', default=ALL_SERIES)
    parser.add_argument('--array-format', default='zarr',
        help='The SLF array format.')
    parser.add_argument('--clevel', type=int, default=7,
        help='Zstd ompression level if array format is zarr.')
    parser.add_argument('--annotation-format', default='json',
        help='The SLF annotation format.')
    
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    assert set(args.series).issubset(set(ALL_SERIES)), f'Series {set(args.series) - set(ALL_SERIES)} not in {ALL_SERIES}'
    logger.info(f'STAGES conversion args: {vars(args)}')
    convert_dataset(**vars(args))
    logger.info('Conversion done.')

import argparse
import logging

from sleeplab_format import writer
from pathlib import Path
from sleeplab_format.models import *
from sleeplab_converters import edf

logger = logging.getLogger(__name__)

def parse_samplearrays(s_load_funcs, sig_headers, header):
    """Read the start_ts and SampleArrays from the EDF."""
    def _parse_samplearray(
            _load_func: Callable[[], np.array],
            _header: dict[str, Any]) -> SampleArray:
        array_attributes = ArrayAttributes(
            # Replace '/' with '_' to avoid errors in filepaths
            name=_header['label'].replace('/', '_').replace('\s', '_'),
            start_ts=start_ts,
            sampling_rate=_header['sample_frequency'],
            unit=_header['dimension']
        )
        return SampleArray(attributes=array_attributes, values_func=_load_func)

    start_ts = header['startdate']
    sample_arrays = {}
    for s_load_func, s_header in zip(s_load_funcs, sig_headers):
        sample_array = _parse_samplearray(s_load_func, s_header)
        sample_arrays[sample_array.attributes.name] = sample_array

    return start_ts, sample_arrays



def parse_edf(edf_path: Path) -> Subject:

    sig_load_funcs, sig_headers, header = edf.read_edf_export(edf_path, annotations=True)

    start_ts, sample_arrays = parse_samplearrays(sig_load_funcs, sig_headers, header)



    metadata = SubjectMetadata(
        subject_id = edf_path.stem,
        recording_start_ts = start_ts)

    return Subject(metadata = metadata,
        sample_arrays = sample_arrays)

def read_data(
        src_dir: Path,
        ds_name: str,
        series_name: str) -> Dataset:
    """Read data from `edf file` and parse to sleeplab Dataset."""
    subjects = {}
    for edf_path in src_dir.iterdir():
        logger.info(f'Start parsing subject {edf_path.name}')
        subject = parse_edf(edf_path)
        subjects[subject.metadata.subject_id] = subject
    
    series = {series_name: Series(name=series_name, subjects=subjects)}
    dataset = Dataset(name=ds_name, series=series)
    
    return dataset


def convert_dataset(
        src_dir: Path,
        dst_dir: Path,
        ds_name: str,
        series_name: str) -> None:
    logger.info(f'Converting Nox cmd tool exported data from {src_dir} to {dst_dir}...')
    logger.info(f'Start reading the data from {src_dir}...')
    dataset = read_data(
        src_dir, ds_name, series_name)
        

    logger.info(f'Start writing the data to {dst_dir}...')
    writer.write_dataset(dataset, dst_dir)
    logger.info(f'Done.')


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=Path, required=True)
    parser.add_argument('--dst_dir', type=Path, required=True)
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--series_name', type=str, required=True)

    return parser


def cli_convert_dataset() -> None:
    parser = create_parser()
    args = parser.parse_args()
    logger.info(f'Nox cmd export tool edf conversion args: {vars(args)}')
    convert_dataset(**vars(args))


if __name__ == '__main__':
    cli_convert_dataset()
"""Parse Profusion export data and write to sleeplab format."""
import argparse
import logging
import re
import xmltodict

from sleeplab_converters import edf
from sleeplab_format import writer
from sleeplab_format.models import *
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger(__name__)


def parse_annotation(d: dict[str, Any], start_ts: datetime) -> Annotation[AASMEvent]:
    name = d.pop('Name')
    start_sec = float(d.pop('Start'))
    duration = float(d.pop('Duration'))
    _start_ts = start_ts + timedelta(seconds=start_sec)
    input_channel = d.pop('Input')
    
    # Parse the rest as extra_attributes
    if len(d) > 0:
        extra_attributes = d
    else:
        extra_attributes = None

    profusion_aasm_event_map = {
        'Unsure': AASMEvent.UNSURE,

        # Score all artifacts as ARTIFACT
        'SpO2 artifact': AASMEvent.ARTIFACT,
        'TcCO2 artifact': AASMEvent.ARTIFACT,
        'ECG Artifact': AASMEvent.ARTIFACT,

        'Arousal (ARO RES)': AASMEvent.AROUSAL_RES,
        'Arousal (ARO SPONT)': AASMEvent.AROUSAL_SPONT,
        'Arousal (ARO PLM)': AASMEvent.AROUSAL_PLM,
        'Arousal (ARO Limb)': AASMEvent.AROUSAL_LM,
        'RERA': AASMEvent.RERA,

        'Limb Movement (Left)': AASMEvent.LM_LEFT,
        'Limb Movement (Right)': AASMEvent.LM_RIGHT,
        'PLM (Left)': AASMEvent.PLM_LEFT,
        'PLM (Right)': AASMEvent.PLM_RIGHT,

        'Central Apnea': AASMEvent.APNEA_CENTRAL,
        'Obstructive Apnea': AASMEvent.APNEA_OBSTRUCTIVE,
        'Mixed Apnea': AASMEvent.APNEA_MIXED,
        'Hypopnea': AASMEvent.HYPOPNEA,
        'SpO2 desaturation': AASMEvent.SPO2_DESAT
    }

    return Annotation[AASMEvent](
        name=profusion_aasm_event_map[name],
        start_ts=_start_ts,
        start_sec=start_sec,
        duration=duration,
        input_channel=input_channel,
        extra_attributes=extra_attributes
    )


def parse_xml(
        xml_path: Path,
        start_ts: datetime,
        # TODO: more precise definition of the scorer?
        scorer: str = 'manual'
        ) -> tuple[dict[str, list[Annotation]], list[int], int]:
    """Read the events and hypnogram from the xml event file."""
    with open(xml_path, 'rb') as f:
        xml_parsed = xmltodict.parse(f)

    event_section = xml_parsed['CMPStudyConfig']['ScoredEvents']
    
    if event_section is not None:
        xml_events = event_section['ScoredEvent']
        events = []
    
        for e in xml_events:
            events.append(parse_annotation(e, start_ts))

        annotations = {f'{scorer}_aasmevents': AASMEvents(scorer=scorer, type='aasmevents', annotations=events)}
    else:
        annotations = None

    hg_section = xml_parsed['CMPStudyConfig']['SleepStages'] 
    if hg_section is not None:
        hypnogram = hg_section['SleepStage']
        hypnogram = [int(stage) for stage in hypnogram]
    else:
        hypnogram = None

    epoch_sec = int(xml_parsed['CMPStudyConfig']['EpochLength'])

    return annotations, hypnogram, epoch_sec


def parse_edf(edf_path: Path) -> tuple[datetime, dict[str, SampleArray]]:
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

    s_load_funcs, s_headers, header = edf.read_edf_export(edf_path)

    start_ts = header['startdate']
    sample_arrays = {}
    for s_load_func, s_header in zip(s_load_funcs, s_headers):
        sample_array = _parse_samplearray(s_load_func, s_header)
        sample_arrays[sample_array.attributes.name] = sample_array

    return start_ts, sample_arrays


def parse_sleep_stage(
        stage_int: int,
        start_ts: datetime,
        epoch: int,
        epoch_sec: float) -> Annotation[AASMSleepStage]:
    stage_map = {
        0: AASMSleepStage.W,
        1: AASMSleepStage.N1,
        2: AASMSleepStage.N2,
        3: AASMSleepStage.N3,
        5: AASMSleepStage.R,
        'U': AASMSleepStage.UNSCORED,
        '?': AASMSleepStage.UNSURE,
        9: AASMSleepStage.ARTIFACT
    }
    
    return Annotation[AASMSleepStage](
        name=stage_map[stage_int],
        start_ts=start_ts + timedelta(seconds=epoch*epoch_sec),
        start_sec=epoch*epoch_sec,
        duration=epoch_sec
    )


def parse_hypnogram(
        start_ts: datetime,
        hg_int: list[int],
        epoch_sec=30,
        scorer='manual') -> Hypnogram:
    sleep_stages = []
    for epoch, stage_int in enumerate(hg_int):
        sleep_stages.append(
            parse_sleep_stage(stage_int, start_ts, epoch, epoch_sec)
        )

    return Hypnogram(scorer=scorer, type='hypnogram', annotations=sleep_stages)


def parse_subject(subject_edf_path: Path) -> Subject:
    subject_id = ''.join(subject_edf_path.stem.split())
    start_ts, sample_arrays = parse_edf(subject_edf_path)
    xml_path = subject_edf_path.parent / (subject_edf_path.name + '.XML')
    annotations, hg_int, epoch_sec = parse_xml(xml_path=xml_path, start_ts=start_ts)
    
    if annotations is None:
        annotations = {}

    if hg_int is not None:
        hypnogram = parse_hypnogram(
            start_ts,
            hg_int,
            epoch_sec=epoch_sec
        )
        
        if hypnogram is not None:
            annotations['manual_hypnogram'] = hypnogram

    metadata = SubjectMetadata(
        subject_id=subject_id,
        recording_start_ts=start_ts
    )

    return Subject(
        metadata=metadata,
        sample_arrays=sample_arrays,
        annotations=annotations
    )


def read_data(
        src_dir: Path,
        ds_name: str,
        series_name: str,
        ) -> Dataset:
    """Read data from `basedir` and parse to sleeplab Dataset."""
    subjects = {}
    for subject_edf_path in src_dir.glob('*.edf'):
        logger.info(f'Start parsing subject {subject_edf_path.name}')
        subject = parse_subject(subject_edf_path)
        subjects[subject.metadata.subject_id] = subject

    series = {series_name: Series(name=series_name, subjects=subjects)}
    dataset = Dataset(name=ds_name, series=series)
    
    return dataset


def convert_dataset(
        src_dir: Path,
        dst_dir: Path,
        ds_name: str,
        series_name: str,
        array_format: str = 'numpy',
        clevel: int = 9) -> None:
    logger.info(f'Converting Profusion data from {src_dir} to {dst_dir}...')
    logger.info(f'Start reading the data from {src_dir}...')
    dataset = read_data(
        src_dir, ds_name, series_name,
        )

    logger.info(f'Start writing the data to {dst_dir}...')
    writer.write_dataset(dataset, dst_dir, array_format=array_format, compression_level=clevel)
    logger.info(f'Done.')


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=Path, required=True)
    parser.add_argument('--dst_dir', type=Path, required=True)
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--series_name', type=str, required=True)
    parser.add_argument('--array_format', default='numpy',
        help='Saving format for numerical arrays. `zarr` or `numpy`')
    parser.add_argument('--clevel', type=int, default=9,
        help='Compression level if array format is zarr.')

    return parser


def cli_convert_dataset() -> None:
    parser = create_parser()
    args = parser.parse_args()
    logger.info(f'Profusion conversion args: {vars(args)}')
    convert_dataset(**vars(args))



if __name__ == '__main__':
    cli_convert_dataset()

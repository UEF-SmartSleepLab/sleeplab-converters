import argparse
import logging
import os

from sleeplab_format import writer
from pathlib import Path
from sleeplab_format.models import *
from sleeplab_converters import edf
from datetime import timedelta

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

def parse_sleep_stage(event, start_rec) -> SleepStageAnnotation:

    stage_str = event[2]

    stage_map = {
        'sleep-wake,Manual': SleepStage.WAKE,
        'sleep-n1,Manual': SleepStage.N1,
        'sleep-n2,Manual': SleepStage.N2,
        'sleep-n3,Manual': SleepStage.N3,
        'sleep-rem,Manual': SleepStage.REM,
    }

    return SleepStageAnnotation(
        name = stage_map[stage_str],
        start_ts = start_rec + timedelta(seconds=event[0]),
        start_sec = event[0],
        duration = event[1]
        )

def parse_for_aasm_annotation(event, start_rec) -> AASMAnnotation:

        #ToDo:  How to detect AASM event? compare to event map?

    name = event[2]
    _start_ts = start_rec + timedelta(seconds=event[0])
    start_sec = event[0]
    duration = event[1]

    nox_edf_aasm_event_map = {
        'Unsure': AASMEvent.UNSURE,

        # Score all artifacts as ARTIFACT
        'signal-artifact,Manual': AASMEvent.ARTIFACT,
        'signal-invalid,Manual': AASMEvent.ARTIFACT,
        'ECG Artifact': AASMEvent.ARTIFACT,

        'plm,Manual': AASMEvent.PLM,

        'arousal,Manual': AASMEvent.AROUSAL,
        'arousal-respiratory,Manual': AASMEvent.AROUSAL_RES,
        'arousal-spontaneous,Manual': AASMEvent.AROUSAL_SPONT,
        'arousal-plm,Manual': AASMEvent.AROUSAL_PLM,
        'arousal-limbmovement,Manual': AASMEvent.AROUSAL_LM,
        'rera,Manual': AASMEvent.RERA,

        'apnea-central,Manual': AASMEvent.APNEA_CENTRAL,
        'apnea-obstructive,Manual': AASMEvent.APNEA_OBSTRUCTIVE,
        'apnea-mixed,Manual': AASMEvent.APNEA_MIXED,
        'hypopnea,Manual': AASMEvent.HYPOPNEA,
        'hypopnea-central,Manual': AASMEvent.HYPOPNEA_CENTRAL,
        'hypopnea-obstructive,Manual': AASMEvent.HYPOPNEA_OBSTRUCTIVE,
        'oxygensaturation-drop,Manual': AASMEvent.SPO2_DESAT,
        'snorebreath,Manual':AASMEvent.SNORE,
        'snore-train,Manual':AASMEvent.SNORE

    }
        
    if name in nox_edf_aasm_event_map.keys():
        return AASMAnnotation(
        name=nox_edf_aasm_event_map[name],
        start_ts=_start_ts,
        start_sec=start_sec,
        duration=duration
    )
    else:
        return None



def parse_annotations(header) -> dict[str, list[Annotation]]:

    events = []
    sleep_stages = []
    AASMevents = []
    st_rec = header['startdate']

    for event in header['annotations']:
        events.append(Annotation(name = event[2], start_ts= st_rec + timedelta(seconds = event[0]), start_sec=event[0], duration=event[1]))

        if event[2][0:5] == 'sleep' and event[2][-6:] == 'Manual':
            sleep_stages.append(parse_sleep_stage(event, start_rec=st_rec))
        
        AASMevent = parse_for_aasm_annotation(event, start_rec=st_rec) 

        if AASMevent is not None:
            AASMevents.append(AASMevent)

    annotations = {
    'all_events': Annotations(annotations = events), 
    'sleep_stages': Hypnogram(annotations = sleep_stages, scorer='Manual'),
    'AASM_events': AASMAnnotations(annotations = AASMevents, scorer='Manual')
    }

    return annotations

def parse_edf(edf_path: Path) -> Subject:

    if os.path.isdir(edf_path):
        sig_load_funcs, sig_headers, header = edf.read_edf_export(edf_path.joinpath('edf_signals.edf'), annotations=True)
    else:
        sig_load_funcs, sig_headers, header = edf.read_edf_export(edf_path, annotations=True)
    

    start_ts, sample_arrays = parse_samplearrays(sig_load_funcs, sig_headers, header)

    annotations = parse_annotations(header)

    metadata = SubjectMetadata(
        subject_id = edf_path.stem,
        recording_start_ts = start_ts)

    return Subject(metadata = metadata,
        sample_arrays = sample_arrays,
        annotations=annotations)

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
    logger.info(f'Converting Nox cli tool exported edfs from {src_dir} to {dst_dir}...')
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
    logger.info(f'Nox cli export tool edf conversion args: {vars(args)}')
    convert_dataset(**vars(args))


if __name__ == '__main__':
    cli_convert_dataset()
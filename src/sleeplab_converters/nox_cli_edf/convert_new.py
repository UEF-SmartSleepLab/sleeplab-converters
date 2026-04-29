import argparse
import logging
import numpy as np
import sleeplab_format as slf
import pandas as pd

from datetime import datetime
from pathlib import Path
from sleeplab_converters import edf
from typing import Any, Callable

logger = logging.getLogger(__name__)


def parse_aasmevent(event, start_ts: datetime, start_sec: float, duration: float) -> slf.models.Annotation:
    event_map = {
        'Artifact': slf.models.AASMEvent.ARTIFACT,

        # Movement events
        'PLM': slf.models.AASMEvent.PLM,
        'PLMS': slf.models.AASMEvent.PLM,
        'LM': slf.models.AASMEvent.LM,

        'Mixed': slf.models.AASMEvent.BRUXISM,
        'Tonic': slf.models.AASMEvent.BRUXISM,
        'Phasic': slf.models.AASMEvent.BRUXISM,

        # Arousal events
        'Arousal': slf.models.AASMEvent.AROUSAL,
        'Spontaneous Arousal': slf.models.AASMEvent.AROUSAL_SPONT,
        'Bruxism Arousal': slf.models.AASMEvent.AROUSAL,
        'Respiratory arousal': slf.models.AASMEvent.AROUSAL_RES,
        'LM Arousal': slf.models.AASMEvent.AROUSAL_LM,
        'PLM Arousal': slf.models.AASMEvent.AROUSAL_PLM,

        'Rera': slf.models.AASMEvent.RERA,

        # Respiratory events
        'A. Central': slf.models.AASMEvent.APNEA_CENTRAL,
        'A. Mixed': slf.models.AASMEvent.APNEA_MIXED,
        'A. Obstructive': slf.models.AASMEvent.APNEA_OBSTRUCTIVE,
        'Apnea': slf.models.AASMEvent.APNEA,
        'Hypopnea': slf.models.AASMEvent.HYPOPNEA,
        'H. Central': slf.models.AASMEvent.HYPOPNEA,
        'H. Obstructive': slf.models.AASMEvent.HYPOPNEA,

        'Desat': slf.models.AASMEvent.SPO2_DESAT,

        'Single Snore': slf.models.AASMEvent.SNORE,
        'Snore Train': slf.models.AASMEvent.SNORE
    }

    if event in event_map.keys():
        return slf.models.Annotation[slf.models.AASMEvent](
            name=event_map[event],
            start_ts=start_ts,
            start_sec=start_sec,
            duration=duration)
    else:
        return None


def parse_sleep_stage(event, start_ts: datetime, start_sec: float, duration: float) -> slf.models.Annotation:
    stage_map = {
        'Wake': slf.models.AASMSleepStage.W,
        'N1': slf.models.AASMSleepStage.N1,
        'N2': slf.models.AASMSleepStage.N2,
        'N3': slf.models.AASMSleepStage.N3,
        'REM': slf.models.AASMSleepStage.R,
        'NREM': slf.models.AASMSleepStage.UNSURE
    }

    if event in stage_map.keys():
        return slf.models.Annotation[slf.models.AASMSleepStage](
            name = stage_map[event],
            start_ts = start_ts,
            start_sec = start_sec,
            duration = duration)
    else:
        return None


def parse_edf(edfpath: Path) -> tuple[datetime, dict[str, slf.models.SampleArray], dict[str, Any]]:
    """Read the start_ts and SampleArrays from the EDF."""
    def _parse_samplearray(
            _load_func: Callable[[], np.array],
            _header: dict[str, Any]) -> slf.models.SampleArray:
        array_attributes = slf.models.ArrayAttributes(
            # Replace '/' and space with '_' to avoid errors in filepaths
            name=_header['label'].replace('/', '_').replace('\s', '_'),
            start_ts=start_ts,
            sampling_rate=_header['sample_frequency'],
            unit=_header['dimension']
        )
        return slf.models.SampleArray(attributes=array_attributes, values_func=_load_func)

    s_load_funcs, s_headers, header = edf.read_edf_export(edfpath)

    start_ts = header['startdate']
    sample_arrays = {}
    for s_load_func, s_header in zip(s_load_funcs, s_headers):
        sample_array = _parse_samplearray(s_load_func, s_header)
        sample_arrays[sample_array.attributes.name] = sample_array

    return start_ts, sample_arrays


def parse_xls(xls_path: Path, rec_start_ts: datetime, scorer: str = 'manual') -> dict[str, slf.models.Annotations]:
    """Read the events and hypnogram from an XLS file."""
    xls_events = pd.read_excel(xls_path)

    start_idx = 0
    if xls_events.Event[0] == '[]':
        xls_events = xls_events.drop(0)
        start_idx = 1

    events = []
    hypnogram = []
    AASMevents = []
    for idx, event in enumerate(xls_events['Event'], start = start_idx):
        start_ts = xls_events['Start Time'][idx]
        start_sec = (start_ts - rec_start_ts).total_seconds()
        duration = xls_events['Duration'][idx]
        
        events.append(slf.models.Annotation[str](name = event, start_ts = start_ts, start_sec = start_sec, duration=duration))

        sleepstage = parse_sleep_stage(event, start_ts=start_ts, start_sec=start_sec, duration=duration)
        if sleepstage is not None:
            hypnogram.append(sleepstage)
        else:
            AASMevent = parse_aasmevent(event, start_ts=start_ts, start_sec=start_sec, duration=duration)
            if AASMevent is not None:
                AASMevents.append(AASMevent)
            
    annotations = {
        f'{scorer}_annotations': slf.models.Annotations(scorer=scorer, annotations=events),
        f'{scorer}_aasmevents': slf.models.AASMEvents(scorer=scorer, annotations=AASMevents),
        f'{scorer}_hypnogram': slf.models.Hypnogram(scorer=scorer, annotations=hypnogram)
    }

    return annotations


def convert_series(src_dir: Path, series_name: str) -> slf.models.Series:
    subjects = {}
    for subject_path in src_dir.iterdir():
        subject_id = subject_path.name
        logger.info(f'Parsing subject {subject_id}')

        edf_path = subject_path.joinpath('edf_signals.edf')
        xls_path = subject_path.joinpath('xls_events.xls')
        
        rec_start_ts, sample_arrays = parse_edf(edfpath=edf_path)
        annotations = parse_xls(xls_path=xls_path, rec_start_ts=rec_start_ts)

        metadata = slf.models.SubjectMetadata(
            subject_id=subject_id,
            recording_start_ts=rec_start_ts,
        )

        subject = slf.models.Subject(
            metadata=metadata,
            sample_arrays=sample_arrays,
            annotations=annotations
        )

        subjects[subject_id] = subject

    series = slf.models.Series(name=series_name, subjects=subjects) 

    return series


def convert_dataset(
        src_dir: Path,
        dst_dir: Path,
        ds_name: str,
        series_name: str,
        array_format: str,
        annotation_format: str) -> None:

    logger.info(f'Reading data from {src_dir}...')
    series = convert_series(src_dir, series_name)

    logger.info(f'Writing data {ds_name} to {dst_dir}')
    dataset = slf.models.Dataset(name=ds_name, series={series_name: series})
    dst_dir.mkdir(parents=True, exist_ok=True)
    slf.writer.write_dataset(
        dataset=dataset,
        basedir=dst_dir,
        annotation_format=annotation_format,
        array_format=array_format
    )


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', type=Path, required=True,
        help='The root folder of the dataset, containing the individual folders of each subject.')
    parser.add_argument('--dst-dir', type=Path, required=True,
        help='The root folder where the SLF dataset is saved.')
    parser.add_argument('--ds-name', default='SLF-converted', help='The name of the SLF dataset created.')
    parser.add_argument('--series-name', default='psg', help='The series name for the PSG recordings.')
    parser.add_argument('--array-format', default='zarr', help='The SLF array format.')
    parser.add_argument('--annotation-format', default='json', help='The SLF annotation format.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    logger.info(f'Converting EDF-signals and XLS-annotations exported from Noxturnal into sleeplab-format.')
    convert_dataset(**vars(args))
    logger.info('Conversion done.')

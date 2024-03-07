import argparse
import json
import logging
import numpy as np
import csv

from datetime import datetime, time, timedelta
from pathlib import Path
from sleeplab_converters import edf
from sleeplab_format import models, writer
from typing import Any, Callable


logger = logging.getLogger(__name__)


def resolve_event_start_ts(start_ts: datetime, start_str: str) -> datetime:
    """Convert time of the day to datetime based on start_ts datetime.

    This function assumes that time_str presents a time of the day within
    24 hours from start_ts.
    """
    start_time = datetime.strptime(start_str, '%H:%M:%S').time()
    _date = start_ts.date()
    if start_time < start_ts.time():
        # If the time is smaller, it belongs to next day
        _date = _date + timedelta(days=1)

    return datetime(
        year=_date.year,
        month=_date.month,
        day=_date.day,
        hour=start_time.hour,
        minute=start_time.minute,
        second=start_time.second,
    )


def parse_csv(csvpath: Path, start_ts: datetime, epoch_sec=30.0) -> tuple[
        list[models.Annotation[models.AASMEvent]],  # Events
        list[models.Annotation[models.AASMSleepStage]],  # Hypnogram
        list[models.Annotation[str]],  # Other annotations
        datetime | None,  # Analysis start
        datetime | None,  # Analysis end
        datetime | None,  # Lights off
        datetime | None]:  # Lights on
    stage_map = {
        'Wake': models.AASMSleepStage.W,
        'Stage1': models.AASMSleepStage.N1,
        'Stage2': models.AASMSleepStage.N2,
        'Stage3': models.AASMSleepStage.N3,
        'REM': models.AASMSleepStage.R,
        'UnknownStage': models.AASMSleepStage.UNSURE
    }
    event_map = {
        'Arousal': models.AASMEvent.AROUSAL,
        'Arousal w/ PLM': models.AASMEvent.AROUSAL_PLM,
        'Arousal w/ Respiratory': models.AASMEvent.AROUSAL_RES,
        'Both Leg w/ Arousal': models.AASMEvent.AROUSAL_LM,
        'Right Leg w/ Arousal': models.AASMEvent.AROUSAL_LM,
        'Left Leg w/ Arousal': models.AASMEvent.AROUSAL_LM,
        'ObstructiveApnea': models.AASMEvent.APNEA_OBSTRUCTIVE,
        'CentralApnea': models.AASMEvent.APNEA_CENTRAL,
        'MixedApnea': models.AASMEvent.APNEA_MIXED,
        'Hypopnea': models.AASMEvent.HYPOPNEA,
        'Desaturation': models.AASMEvent.SPO2_DESAT,
        'Desaturation w/ Respiratory': models.AASMEvent.SPO2_DESAT,
        'Snore': models.AASMEvent.SNORE,
        'RERA': models.AASMEvent.RERA,
        'Left Leg': models.AASMEvent.LM_LEFT,
        'Right Leg': models.AASMEvent.LM_RIGHT,
        'Both Leg': models.AASMEvent.LM,
        'PLM': models.AASMEvent.PLM,
    }

    hg = []
    events = []
    other_annotations = []
    cal_times = []

    lights_off = None
    lights_on = None

    with open(csvpath, 'r') as f:
        reader = csv.reader(f, delimiter=',', skipinitialspace=True)
        _ = next(reader, None)  # Skip the header line
        for row in reader:
            if len(row) > 3:
                logger.info(f'Over 3 commas on row, concatenating {row}...')
                row = row[:2] + [', '.join(row[2:])]
                logger.info(f'...to {row}')
            start_time_str, duration_str, name_str = row
            name_str = name_str.strip()
            event_start_ts = resolve_event_start_ts(start_ts, start_time_str)
            duration = float(duration_str)

            if name_str in stage_map.keys():
                hg.append(models.Annotation[models.AASMSleepStage](
                    name=stage_map[name_str],
                    start_ts=event_start_ts,
                    start_sec=(event_start_ts - start_ts).total_seconds(),
                    duration=epoch_sec if duration == 0.0 else duration
                ))
            elif name_str in event_map.keys():
                events.append(models.Annotation[models.AASMEvent](
                    name=event_map[name_str],
                    start_ts=event_start_ts,
                    start_sec=(event_start_ts - start_ts).total_seconds(),
                    duration=duration
                ))
            elif name_str == 'Cal':
                # Interpret the 'cal' events as analysis_start and analysis_end
                cal_times.append(event_start_ts)
            elif (name_str in ('LightsOff',)
                  or name_str.lower().startswith('l/o')
                  or name_str.lower().startswith('lights out')
                  or name_str.lower().startswith('lights off')):
                if lights_off is not None:
                    logger.warn('lights_off already assigned')
                else:
                    lights_off = event_start_ts
            elif (name_str in ('LightsOn', 'LON')
                  or name_str.lower().startswith('lights on')
                  or name_str.lower().startswith('lights oon')):
                if lights_on is not None:
                    logger.warn('lights_on already assigned')
                else:
                    lights_on = event_start_ts
            else:
                other_annotations.append(models.Annotation[str](
                    name=name_str,
                    start_ts=event_start_ts,
                    start_sec=(event_start_ts - start_ts).total_seconds(),
                    duration=duration
                ))

    if len(cal_times) == 2:
        analysis_start = cal_times[0]
        analysis_end = cal_times[1]
    else:
        if len(cal_times) > 0:
            logger.info(f'There should be exactly 2 Cal times, but got {cal_times}')
        analysis_start = None
        analysis_end = None

    # Set lights off and on to analysis_start and end, or start and end of the scored hypnogram,
    # if annotations for them were not found.
    if lights_off is None:
        lights_off = analysis_start or hg[0].start_ts
    if lights_on is None:
        lights_on = analysis_end or hg[-1].start_ts + timedelta(seconds=hg[-1].duration)

    return events, hg, other_annotations, analysis_start, analysis_end, lights_off, lights_on


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


def convert_series(src_dir: Path, series_name: str) -> models.Series:
    subjects = {}
    error_counts = {
        'underscore_in_name': 0,
        'EDF_does_not_exist': 0,
        'CSV_parse_error': 0
    }
    # Loop through csv files
    for csvpath in (src_dir / 'original' / 'STAGES PSGs' / series_name).glob('*.csv'):
        logger.info(f'Parsing subject {csvpath.stem}...')
        edfpath = csvpath.parent / f'{csvpath.stem}.edf'
        subject_id = csvpath.stem
        if '_' in edfpath.stem:
            logger.info(f'Skipping subject with underscore in name (probably it is MSLT): {subject_id}')
            error_counts['underscore_in_name'] += 1
            continue
        # If there is no edf corresponding to csv, skip
        if not edfpath.exists():
            logger.info(f'EDF file matching the CSV does not exist: {edfpath}')
            error_counts['EDF_does_not_exist'] += 1
            continue

        start_ts, sample_arrays = parse_edf(edf_path=edfpath)
        try:
            events, hypnogram, other_ann, analysis_start, analysis_end, lights_off, lights_on = parse_csv(
                csvpath=csvpath,
                start_ts=start_ts
            )
        except Exception as e:
            logger.warn(f'Skipping subject {subject_id} due to error in CSV parsing:')
            logger.warn(e)
            error_counts['CSV_parse_error'] += 1
            continue

        annotations = {
            'STAGES_aasmevents': models.AASMEvents(scorer='STAGES', annotations=events),
            'STAGES_hypnogram': models.Hypnogram(scorer='STAGES', annotations=hypnogram),
            'STAGES_annotations': models.Annotations(scorer='STAGES', annotations=other_ann)
        }
        metadata = models.SubjectMetadata(
            subject_id=subject_id,
            recording_start_ts=start_ts,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            lights_off=lights_off,
            lights_on=lights_on
        )
        subject = models.Subject(
            metadata=metadata,
            sample_arrays=sample_arrays,
            annotations=annotations
        )

        subjects[subject_id] = subject

    series = models.Series(name=series_name, subjects=subjects)
    
    return series, error_counts


ALL_SERIES = [
    'BOGN',
    'GSBB',
    'GSDV',
    'GSLH',
    'GSSA',
    'GSSW',
    'MSMI',
    'MSNF',
    'MSQW',
    'MSTH',
    'MSTR',
    'STLK',
    'STNF'
]


def convert_dataset(
        src_dir: Path,
        dst_dir: Path,
        ds_name: str = 'STAGES',
        series: list[str] = ALL_SERIES,
        array_format: str = 'zarr',
        clevel: int = 7,
        annotation_format: str = 'json') -> None:
    series_dict = {}
    all_error_counts = {}
    for series_name in series:
        logger.info(f'Converting series {series_name}...')
        _series, _error_counts = convert_series(
            src_dir=src_dir,
            series_name=series_name
        )
        series_dict[series_name] = _series
        all_error_counts[series_name] = _error_counts

    error_count_path = dst_dir / ds_name / 'conversion_error_counts.json'
    logger.info(f'Writing error counts to {error_count_path}')
    with open(error_count_path, 'w') as f:
        json.dump(all_error_counts, f, indent=4)

    dataset = models.Dataset(name=ds_name, series=series_dict)
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

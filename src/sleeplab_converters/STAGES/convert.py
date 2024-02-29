import argparse
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
        'CentralApnea': models.AASMEvent.APNEA_CENTRAL,
        'Desaturation': models.AASMEvent.SPO2_DESAT,
        'Hypopnea': models.AASMEvent.HYPOPNEA,
        'MixedApnea': models.AASMEvent.APNEA_MIXED,
        'ObstructiveApnea': models.AASMEvent.APNEA_OBSTRUCTIVE,
        'Snore': models.AASMEvent.SNORE
    }
    skip_names = [
        'Awake',
        'CAw',
        'CSBr',
        'MChg',
        'MA',
        'MAa',
        'mt',
        'MT',
        'mta',
        'MTa',
        'MTw',
        'mtw',
        'Nasal Breathing',
        'OAw',
        'OHw',
        'Oral Breathing',
        'Per Slp',
        'REM Aw',
        'UD1',
        'UD1a',
        'UD1w',
        'UD2',
        'UD2a',
        'UD2w',
        'UDAr',
        'UDAra',
        'Unknown',

        'BIO-CALS',
        'Lie quietly with eyes open',
        'Lie quietly with eyes closed',
        'Look up and down 5 times',
        'Look left and right 5 times',
        'Blink 5 times',
        'Grit teeth for 5 seconds',
        'Flex left foot 5 times',
        'Flex right foot 5 times',
        'Breathe through nose only for 30 seconds',
        'Breathe through mouth only for 30 seconds',
        'Hold breath for 10 seconds',
        'Cough',
        'COUGH',

        'Eyes Closed',
        'EYES CLOSED',
        'Eyes Open',
        'EYES OPEN',
        'Look Up Down',
        'EYES UP AND DOWN',
        'Look Left Right',
        'EYES LEFT/RIGHT',
        'Blink 5 Times',
        'BLINK',
        'Grit Teeth',
        'GRIT TEETH',
        'Flex Left Leg',
        'FLXL',
        'Flex Right Leg',
        'FLXR',
        'NOSE',
        'MOUTH',
        'HOLD BREATH',

        'MOVED LEFT',
        'MOVED RIGHT',
        'MOVED RIGHT SIDE',
        'MOVED SUPINE',
        'LEFT SIDE',
        'Supine',

        'EMERGENCY INTERVENTION',
        'TIR- START CPAP',
        'TIR TO ADJUST ORAL FLOW SENSOR',
        'TIR TO REATTATCH HEAD LEADS AFTER PT CALLED FOR TECH TO COME REPLACE THEM',
        'TIR TO CHECK PULSE OX',
        'TIR TO FIX PULSE  OX',
        'TIR FIXING CANNULA',
        'TIR TO HELP PT UP TO THE RESTROOM- CANFLOW FIXED AT THIS TIME',
        'TIR TO TURN TV OFF AS PT FELL ASLEEP WITHIT ON',
        'TIR TO REPOSITION POx ON FINGER.',
        'TIR TO HELP PT UP TO THE RESTROOM',
        'TIR CHECKING FLOWS',
        'TIR GETTING PT SETUP ON CPAP',
        'TOR',
        'TOR. FIXED CHIN',

        'MACHINE CALS',

        "PT'S LOUT DELAYED DUE TO NORMAL SLEEP TIME BEING ABOUT 0000 FOR CLASS HOURS.",

        'Begin PT Cals',
        'TECH AWARE OF M2',
        'PATIENT WORRIED THAT SHE IS NOT SLEEPING',
        'TechIn',
        'PT to Restroom',
        'TechOut',
        'PATIENT STATED SHE IS READY TO END STUDY WHEN ENOUGHT TIME HAS PASSED.',
        '0529-0548 PULSE OX NOT READING ACURATLY DUE TO SLIDING OFF A BIT',
        'CHECK M2',
        'REPLACED LEADS',
        'PATIENT REQUESTING LON',
        'IN TO FIX PULSE OX',
        'TECH JUST REALIZED CPAP PRESSURE WAS SET AT 6CMH20 INSTEAD OF 5CMH20.',
        'TECH DECREASED PRESSURE TO 5CMH20',
        'INCREASED PRESSURE TO 7CMH20 FOR PHYSICIAN COMPARISON',
        'PRESSURE INCREASED TO 9CMH20 FOR PHYSICIAN COMPARISON. TIDAL VOL 404',
    ]

    hg = []
    events = []
    cal_times = []

    lights_off = None
    lights_on = None

    with open(csvpath, 'r') as f:
        reader = csv.reader(f, delimiter=',', skipinitialspace=True)
        _ = next(reader, None)  # Skip the header line
        for row in reader:
            if len(row) != 3:
                logger.info(f'Wrong number of commas on row, skipping: {row}')
                continue
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
            elif name_str in ('LightsOff',):
                assert lights_off is None, 'lights_off already assigned'
                lights_off = event_start_ts
            elif name_str in ('LightsOn', 'LON'):
                assert lights_on is None, 'lights_on already assigned'
                lights_on = event_start_ts
            elif name_str in skip_names:
                continue
            else:
                raise ValueError(f'unknown name_str: {name_str}')

    if len(cal_times) != 2:
        logger.info(f'There should be exactly 2 Cal times, but got {cal_times}')
        analysis_start = None
        analysis_end = None
    else:
        analysis_start = cal_times[0]
        analysis_end = cal_times[1]

    return events, hg, analysis_start, analysis_end, lights_off, lights_on


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
    # Loop through csv files
    for csvpath in (src_dir / 'original' / 'STAGES PSGs' / 'BOGN').glob('*.csv'):
        logger.info(f'Parsing subject {csvpath.stem}...')
        edfpath = csvpath.parent / f'{csvpath.stem}.edf'
        # If there is no edf corresponding to csv, skip
        if not edfpath.exists():
            logger.info(f'EDF file matching the CSV does not exist: {edfpath}')
            continue

        subject_id = csvpath.stem
        start_ts, sample_arrays = parse_edf(edf_path=edfpath)
        events, hypnogram, analysis_start, analysis_end, lights_off, lights_on = parse_csv(
            csvpath=csvpath,
            start_ts=start_ts
        )

        annotations = {
            'STAGES_aasmevents': models.AASMEvents(scorer='STAGES', type='aasmevents', annotations=events),
            'STAGES_hypnogram': models.Hypnogram(scorer='STAGES', type='hypnogram', annotations=hypnogram)
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
    
    return series


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
    'SLTK',
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
    for series_name in series:
        logger.info(f'Converting series {series_name}...')
        series_dict[series_name] = convert_series(
            src_dir=src_dir,
            series_name=series_name
        )

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

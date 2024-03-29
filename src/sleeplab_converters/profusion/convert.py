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


def str_to_time(time_str):
    hour, minute, _second = time_str.split(sep=':')
    hour = int(hour)
    minute = int(minute)
    _second = float(_second)
    second = int(_second)
    microsecond = int(_second % 1 * 1e6)

    return time(hour=hour, minute=minute, second=second, microsecond=microsecond)


def resolve_datetime(start_ts: datetime, _time: datetime) -> datetime:
    """Convert time of the day to datetime based on start_ts datetime.
    
    This function assumes that time_str presents a time of the day within
    24 hours from start_ts.
    """
    _date = start_ts.date()
    if _time < start_ts.time():
        # If the time is smaller, it belongs to next day
        _date = _date + timedelta(days=1)

    return datetime(
        year=_date.year,
        month=_date.month,
        day=_date.day,
        hour=_time.hour,
        minute=_time.minute,
        second=_time.second,
        microsecond=_time.microsecond    
    )
    
    
def resolve_log_start_ts(start_ts: datetime, _time: datetime) -> datetime:
    """
    Start_ts represents the start of recording, and there may be
    logging before that, so create a log_start_ts to be used in
    resolve_datetime().
    """
    # If start_ts is after midnight, and there are logs before midnight,
    # those logs actually happened during the previous date
    if start_ts.hour < 12 and _time.hour > 12:
        _date = start_ts.date() - timedelta(days=1)
    # Need to add 1 day if other way around
    elif start_ts.hour > 12 and _time.hour < 12:
        _date = start_ts.date() + timedelta(days=1)
    # If both are before midnight or after midnight, they belong to same date.
    else:
        _date = start_ts.date()

    return datetime(
        year=_date.year,
        month=_date.month,
        day=_date.day,
        hour=_time.hour,
        minute=_time.minute,
        second=_time.second,
        microsecond=_time.microsecond
    )


def parse_study_logs(log_file_path: Path, start_ts: datetime) -> Logs:
    """Parse txt study logs so that the time will be converted to datetime.
    
    Resolving the datetime relies on the heuristic that logs are for a duration
    shorter than 24h.

    TODO: Add validation for the 24h assumption.
    """
    def _parse_line(l):
        try:
            time_str, _, _, text = l.split(sep=',', maxsplit=3)
        except ValueError:
            logger.warning(f'Could not parse line\n\t{l}\n\tfrom log file {log_file_path}')
            return None
        _time = str_to_time(time_str)
        _start_ts = resolve_datetime(log_start_ts, _time)
        _start_sec = (_start_ts - start_ts).total_seconds()
        return Annotation[str](start_ts=_start_ts, start_sec=_start_sec, name=text)
    
    res = []
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logger.warning(f'Log file not found: {log_file_path}')
        return None
        
    if len(lines) == 0:
        return None
    
    first_log_time_str = lines[0].split(',', maxsplit=1)[0].strip()
    log_start_ts = resolve_log_start_ts(start_ts, str_to_time(first_log_time_str))

    for line in lines:
        l_parsed = _parse_line(line.strip())
        if l_parsed is not None:
            res.append(l_parsed)

    return Logs(scorer='study', type='logs', annotations=res)


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
        scorer: str = 'profusion'
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


def parse_subject_id(idinfo_path: Path) -> str:
    """Parse the profusion study id from txt_idinfo.txt.
    
    The study id is assumed to be contained in the idinfo generated from
    Profusion in the form 'Compumedics ProFusion PSG - [123456 1.1.1900]'.

    The study id is assumed to be a decimal integer.
    """
    with open(idinfo_path, 'r') as f:
        id_info = f.readlines()[0]

    # Match the decimal integer between '[' and space
    re_str = r'\[(\d+)\s'
    # This will raise AttributeError if no match
    subject_id = re.search(re_str, id_info).group(1)
    
    return str(subject_id)


def parse_sleep_stage(
        stage_str: str,
        start_ts: datetime,
        epoch: int,
        epoch_sec: float) -> Annotation[AASMSleepStage]:
    stage_map = {
        'W': AASMSleepStage.W,
        'N1': AASMSleepStage.N1,
        'N2': AASMSleepStage.N2,
        'N3': AASMSleepStage.N3,
        'R': AASMSleepStage.R,
        'U': AASMSleepStage.UNSCORED,
        '?': AASMSleepStage.UNSURE,
        'A': AASMSleepStage.ARTIFACT
    }
    
    return Annotation[AASMSleepStage](
        name=stage_map[stage_str],
        start_ts=start_ts + timedelta(seconds=epoch*epoch_sec),
        start_sec=epoch*epoch_sec,
        duration=epoch_sec
    )


def parse_hypnogram(
        subject_dir: Path,
        start_ts: datetime,
        hg_file: str,
        hg_int: list[int],
        epoch_sec=30,
        scorer='profusion') -> Hypnogram:
    try:
        with open(subject_dir / hg_file) as f:
            hg_str = f.readlines()
    except FileNotFoundError:
        logger.warn(f'Hypnogram not found from {subject_dir / hg_file}, return None')
        return None
    
    hg_str = [l.strip() for l in hg_str]
    _msg = f'Hypnograms from xml and txt files have different lengths ({len(hg_int)} =! {len(hg_str)})'
    assert len(hg_int) == len(hg_str), _msg
    
    # This is not needed anymore since writing hypnograms as annotations
    # hg_map = {int_stage: str_stage for int_stage, str_stage in zip(hg_int, hg_str)}

    sleep_stages = []
    for epoch, stage_str in enumerate(hg_str):
        sleep_stages.append(
            parse_sleep_stage(stage_str, start_ts, epoch, epoch_sec)
        )

    return Hypnogram(scorer=scorer, type='hypnogram', annotations=sleep_stages)


def parse_subject(subject_dir: Path, file_names: dict[str, str]) -> Subject:
    if file_names['idinfo_file'] == '':
        subject_id = subject_dir.name
    else:
        subject_id = parse_subject_id(subject_dir / file_names['idinfo_file'])
    start_ts, sample_arrays = parse_edf(subject_dir / file_names['edf_file'])
    annotations, hg_int, epoch_sec = parse_xml(subject_dir / file_names['xml_file'],
        start_ts=start_ts)
    
    if annotations is None:
        annotations = {}

    if hg_int is not None:
        hypnogram = parse_hypnogram(
            subject_dir,
            start_ts,
            file_names['hg_file'],
            hg_int,
            epoch_sec=epoch_sec
        )
        
        if hypnogram is not None:
            annotations['profusion_hypnogram'] = hypnogram

    study_logs = parse_study_logs(subject_dir / file_names['log_file'], start_ts)
    if study_logs is not None:
        annotations['study_logs'] = study_logs

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
        file_names: dict[str, str]) -> Dataset:
    """Read data from `basedir` and parse to sleeplab Dataset."""
    subjects = {}
    for subject_dir in src_dir.iterdir():
        logger.info(f'Start parsing subject {subject_dir.name}')
        subject = parse_subject(subject_dir, file_names=file_names)
        subjects[subject.metadata.subject_id] = subject

    series = {series_name: Series(name=series_name, subjects=subjects)}
    dataset = Dataset(name=ds_name, series=series)
    
    return dataset


def convert_dataset(
        src_dir: Path,
        dst_dir: Path,
        ds_name: str,
        series_name: str,
        xml_file: str = 'edf_signals.edf.XML',
        log_file: str = 'txt_studylog.txt',
        edf_file: str = 'edf_signals.edf',
        idinfo_file: str = 'txt_idinfo.txt',
        hg_file: str = 'txt_hypnogram.txt',
        array_format: str = 'zarr',
        clevel: int = 9) -> None:
    logger.info(f'Converting Profusion data from {src_dir} to {dst_dir}...')
    logger.info(f'Start reading the data from {src_dir}...')
    dataset = read_data(
        src_dir, ds_name, series_name,
        file_names={
            'xml_file': xml_file,
            'log_file': log_file,
            'edf_file': edf_file,
            'idinfo_file': idinfo_file,
            'hg_file': hg_file
        })

    logger.info(f'Start writing the data to {dst_dir}...')
    writer.write_dataset(dataset, dst_dir, array_format=array_format, compression_level=clevel)
    logger.info(f'Done.')


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=Path, required=True)
    parser.add_argument('--dst_dir', type=Path, required=True)
    parser.add_argument('--ds_name', type=str, required=True)
    parser.add_argument('--series_name', type=str, required=True)
    parser.add_argument('--xml_file', type=str, default='edf_signals.edf.XML')
    parser.add_argument('--log_file', type=str, default='txt_studylog.txt')
    parser.add_argument('--edf_file', type=str, default='edf_signals.edf')
    parser.add_argument('--idinfo_file', type=str, default='txt_idinfo.txt')
    parser.add_argument('--hg_file', type=str, default='txt_hypnogram.txt')
    parser.add_argument('--array_format', default='zarr',
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

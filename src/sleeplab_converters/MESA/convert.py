import argparse
import logging
import numpy as np
import sleeplab_format as slf
import xmltodict

from datetime import datetime, timedelta
from pathlib import Path
from sleeplab_converters import edf
from typing import Any, Callable


logger = logging.getLogger(__name__)


def parse_aasmevent(e_dict: dict[str, Any], rec_start_ts: datetime, rec_duration: float) -> slf.models.Annotation:
    # TODO: add SignalLocation, SpO2 nadir and baseline
    event_map = {
        'SpO2 artifact|SpO2 artifact': slf.models.AASMEvent.ARTIFACT,
        'Respiratory artifact|Respiratory artifact': slf.models.AASMEvent.ARTIFACT,
        'SpO2 desaturation|SpO2 desaturation': slf.models.AASMEvent.SPO2_DESAT,
        'Arousal|Arousal ()': slf.models.AASMEvent.AROUSAL,
        'ASDA arousal|Arousal (ASDA)': slf.models.AASMEvent.AROUSAL,
        'Spontaneous arousal|Arousal (ARO SPONT)': slf.models.AASMEvent.AROUSAL_SPONT,
        'Arousal (ARO Limb)': slf.models.AASMEvent.AROUSAL_LM,
        'Arousal (AASM)': slf.models.AASMEvent.AROUSAL,
        'Respiratory effort related arousal|RERA': slf.models.AASMEvent.RERA,
        'Arousal resulting from respiratory effort|Arousal (ARO RES)': slf.models.AASMEvent.AROUSAL_RES,
        'Limb movement - left|Limb Movement (Left)': slf.models.AASMEvent.LM_LEFT,
        'Periodic leg movement - left|PLM (Left)': slf.models.AASMEvent.PLM_LEFT,

        # In the MESA scorings, 'Hypopnea' has airflow reduction of 30%-50% from baseline,
        # and 'Unsure' is a hypopnea with reduction more than 50%...
        'Hypopnea|Hypopnea': slf.models.AASMEvent.HYPOPNEA,
        'Unsure|Unsure': slf.models.AASMEvent.HYPOPNEA,

        'Obstructive apnea|Obstructive Apnea': slf.models.AASMEvent.APNEA_OBSTRUCTIVE,
        'Central apnea|Central Apnea': slf.models.AASMEvent.APNEA_CENTRAL,
        'Mixed apnea|Mixed Apnea': slf.models.AASMEvent.APNEA_MIXED
    }
    
    name = event_map[e_dict['EventConcept']]
    start_sec = float(e_dict['Start'])
    duration = float(e_dict['Duration'])
    start_ts = rec_start_ts + timedelta(seconds=start_sec)

    _msg = 'Event end is later than the recording end'
    assert start_ts + timedelta(seconds=duration) <= rec_start_ts + timedelta(seconds=rec_duration), _msg

    return slf.models.Annotation[slf.models.AASMEvent](
        name=name,
        start_ts=start_ts,
        start_sec=start_sec,
        duration=duration
    )


def parse_sleep_stage(e_dict: dict[str, Any], rec_start_ts: datetime, rec_duration: float) -> slf.models.Annotation:
    stage_map = {
        'Wake|0': slf.models.AASMSleepStage.W,
        'Stage 1 sleep|1': slf.models.AASMSleepStage.N1,
        'Stage 2 sleep|2': slf.models.AASMSleepStage.N2,
        'Stage 3 sleep|3': slf.models.AASMSleepStage.N3,
        'REM sleep|5': slf.models.AASMSleepStage.R,
        'Unscored|9': slf.models.AASMSleepStage.UNSCORED,
        'Stage 4 sleep|4': slf.models.AASMSleepStage.N3
    }

    name = stage_map[e_dict['EventConcept']]
    start_sec = float(e_dict['Start'])
    duration = float(e_dict['Duration'])
    start_ts = rec_start_ts + timedelta(seconds=start_sec)

    _msg = 'Sleep stage end is later than the recording end'
    assert start_ts + timedelta(seconds=duration) <= rec_start_ts + timedelta(seconds=rec_duration), _msg

    return slf.models.Annotation[slf.models.AASMSleepStage](
        name=name,
        start_ts=start_ts,
        start_sec=start_sec,
        duration=duration
    )


def parse_xml(xmlpath: Path, rec_start_ts: datetime, scorer: str = 'nsrr') -> dict[str, slf.models.Annotations]:
    """Read the events and hypnogram from the XML file."""
    with open(xmlpath, 'rb') as f:
        xml_parsed = xmltodict.parse(f)

    epoch_sec = int(xml_parsed['PSGAnnotation']['EpochLength'])
    hg = []
    events = []
    
    for e in xml_parsed['PSGAnnotation']['ScoredEvents']['ScoredEvent']:
        if e['EventConcept'] == 'Recording Start Time':
            _start_ts = datetime.strptime(e['ClockTime'], '%d.%m.%y %H.%M.%S')
            assert rec_start_ts == _start_ts, f'Start ts from EDF does not match the XML'
            rec_duration = float(e['Duration'])
        elif e['EventType'].startswith('Stages'):
            hg.append(parse_sleep_stage(e, rec_start_ts, rec_duration))
        elif e['EventConcept'] == 'Technician Notes':
            # Some arousals have been scored as technician notes...
            if e['Notes'] in ('Arousal (ARO Limb)', 'Arousal (AASM)'):
                e['EventConcept'] = e['Notes']
                events.append(parse_aasmevent(e, rec_start_ts, rec_duration))
            else:
                logger.info(f'Unknown event: {e}')
                raise KeyError(e['EventConcept'])
        elif e['EventConcept'] in (
                'Narrow complex tachycardia|Narrow Complex Tachycardia',
                'Periodic breathing|Periodic Breathing'):
            # Skip these events for now
            logger.info(f'Skipping event {e}')
            continue
        else:
            # Discard the event on error
            try:
                events.append(parse_aasmevent(e, rec_start_ts, rec_duration))
            except AssertionError as e:
                print(e)

    annotations = {
        f'{scorer}_aasmevents': slf.models.AASMEvents(scorer=scorer, annotations=events),
        f'{scorer}_hypnogram': slf.models.Hypnogram(scorer=scorer, annotations=hg)
    }

    return annotations


def parse_edf(edfpath: Path) -> tuple[datetime, dict[str, slf.models.SampleArray]]:
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


def convert_series(src_dir: Path, series_name: str) -> slf.models.Series:
    subjects = {}
    for edfpath in (src_dir / 'edfs').glob('*.edf'):
        subject_id = edfpath.stem
        logger.info(f'Parsing subject {subject_id}')
        xmlpath = edfpath.parent.parent / 'annotations-events-nsrr' / f'{subject_id}-nsrr.xml'
        if not xmlpath.exists():
            logger.info(f'Event file matching the EDF does not exist: {xmlpath}')
            continue
        
        rec_start_ts, sample_arrays = parse_edf(edfpath=edfpath)
        annotations = parse_xml(xmlpath, rec_start_ts)
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
    logger.info(f'Converting series {series_name}')
    series = convert_series(src_dir / 'polysomnography', series_name)
    dataset = slf.models.Dataset(name=ds_name, series={series_name: series})
    logger.info(f'Writing dataset {ds_name} to {dst_dir}')
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
        help='The root folder of the MESA dataset, containing `polysomnography` folder.')
    parser.add_argument('--dst-dir', type=Path, required=True,
        help='The root folder where the SLF dataset is saved.')
    parser.add_argument('--ds-name', default='MESA', help='The name of the SLF dataset created.')
    parser.add_argument('--series-name', default='psg', help='The series name for the PSG recordings.')
    parser.add_argument('--array-format', default='zarr', help='The SLF array format.')
    parser.add_argument('--annotation-format', default='json', help='The SLF annotation format.')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    logger.info(f'Converting MESA to sleeplab-format. Conversion args: {vars(args)}')
    convert_dataset(**vars(args))
    logger.info('Conversion done.')

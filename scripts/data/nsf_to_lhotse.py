import argparse
from json import loads
import os
from pathlib import Path

from lhotse import CutSet, SupervisionSegment, SupervisionSet, MonoCut, Recording
from lhotse.supervision import AlignmentItem
from tqdm import tqdm


def main(dataset_path, output_dir, output_fname_prefix, word_level=False, create_word_alignment=True):
    meetings = sorted(os.listdir(dataset_path))
    cuts = []

    for meeting in tqdm(meetings):
        meeting_root = dataset_path / meeting
        transcription_path = meeting_root / 'gt_transcription.json'
        devices = sorted(list(filter(lambda x: x != 'close_talk' and os.path.isdir(meeting_root/x), os.listdir(meeting_root))))

        with open(transcription_path, 'r') as f:
            transcription_json = loads(f.read())

        for device in devices:
            # Ignore multi channel.
            if 'mc' in device:
                continue

            recording_path = meeting_root / device / 'ch0.wav'
            recording = Recording.from_file(recording_path)
            recording.id = f'{meeting}_{device}'

            supervisions = []
            for segment in transcription_json:
                speaker_id = segment['speaker_id']
                channel = [0]

                if word_level:
                    for text, start_time, end_time in segment['word_timing']:
                        start_time = float(start_time)
                        end_time = float(end_time)

                        supervisions.append(SupervisionSegment(
                            id=f'{meeting}_{device}_{str(int(start_time*100)).zfill(6)}_{str(int(end_time*100)).zfill(6)}',
                            recording_id=recording.id,
                            start=start_time,
                            duration=end_time - start_time,
                            channel=channel,
                            text=text,
                            speaker=speaker_id
                        ))
                else:
                    start_time = float(segment['start_time'])
                    end_time = float(segment['end_time'])
                    text = segment['text']
                    alignment = None

                    if create_word_alignment:
                        alignment = {
                            'word': []
                        }

                        for alig_text, alig_start_time, alig_end_time in segment['word_timing']:
                            # Skip all the fillings.
                            if '<' in alig_text or '>' in alig_text:
                                continue
                            alig_start_time = float(alig_start_time)
                            alig_end_time = float(alig_end_time)
                            alignment['word'].append(AlignmentItem(symbol=alig_text, start=alig_start_time, duration=alig_end_time - alig_start_time))

                    supervisions.append(SupervisionSegment(
                        id=f'{meeting}_{device}_{str(int(start_time*100)).zfill(6)}_{str(int(end_time*100)).zfill(6)}',
                        recording_id=recording.id,
                        start=start_time,
                        duration=end_time - start_time,
                        channel=channel,
                        text=text,
                        speaker=speaker_id,
                        alignment=alignment
                    ))

            cuts.append(MonoCut(
                id=recording.id,
                start=0,
                duration=recording.duration,
                channel=0,
                supervisions=supervisions,
                recording=recording
            ))

    CutSet.from_cuts(cuts).to_file(f'{output_dir}/{output_fname_prefix}{"_wlevel" if word_level else ""}.jsonl.gz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_fname_prefix', type=str, required=True)
    parser.add_argument('--word_level', action='store_true')
    parser.add_argument('--create_word_alignment', type=bool, default=True)

    args = parser.parse_args()

    main(Path(args.dataset_path), args.output_dir, args.output_fname_prefix, args.word_level, args.create_word_alignment)

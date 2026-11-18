from config import SAMPLE_RATE
from config import SEGMENT_DURATION


def split_audio(audio):

    """
    Split audio into small segments
    """

    segment_length = SAMPLE_RATE * SEGMENT_DURATION

    segments = []

    timestamps = []

    for i in range(0, len(audio), segment_length):

        segment = audio[i:i + segment_length]

        # Keep only full segments
        if len(segment) == segment_length:

            segments.append(segment)

            start_time = i / SAMPLE_RATE
            end_time = (i + segment_length) / SAMPLE_RATE

            timestamps.append(
                (start_time, end_time)
            )

    return segments, timestamps
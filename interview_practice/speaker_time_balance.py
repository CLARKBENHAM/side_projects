# %%
import os

print(len(os.getenv("HUGGING_FACE_TOKEN")))
import csv

from pyannote.audio import Pipeline

# Initialize the pre-trained diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=os.getenv("HUGGING_FACE_TOKEN")
)


# Apply the pipeline to your audio file
diarization = pipeline("temp/audio/Distyl Karime system design.wav")

# Initialize a dictionary to store speaking durations
speaking_times = {}

# Iterate over the diarization segments
max_ix = 0
csv_file_path = "temp/transcripts/distyl_karime_speaker_diarization.csv"
# Open the CSV file in write mode
with open(csv_file_path, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header row
    csv_writer.writerow(["Start Time", "End Time", "Speaker", "Duration", "a"])
    for ix, (segment, a, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        duration = segment.end - segment.start
        csv_writer.writerow([segment.start, segment.end, speaker, duration, a])
        print(ix, segment, speaker, a)
        if speaker in speaking_times:
            speaking_times[speaker] += duration
        else:
            speaking_times[speaker] = duration
        max_ix = ix
print(f"Num turns: {max_ix}")

# Display the speaking times
for speaker, total_time in speaking_times.items():
    print(f"{speaker}: {total_time:.2f} seconds")
# %%

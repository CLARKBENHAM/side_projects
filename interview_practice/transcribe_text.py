import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
import torch

# Directory path, reads from /audio writes to /transcripts
temp_folder = "temp"

# Load Whisper model

# Check if a GPU is available
if torch.cuda.is_available():
    # Load the model on the GPU with FP16 precision
    model = whisper.load_model("base").to(device="cuda", dtype=torch.float16)
    print("Using GPU with FP16 precision.")
else:
    # Load the model on the CPU with FP32 precision
    model = whisper.load_model("base").to(device="cpu", dtype=torch.float32)
    print("Using CPU with FP32 precision.")


# Function to list audio files without transcripts
def list_audio_files_without_transcripts(folder):
    audio_files = []
    for filename in os.listdir(os.path.join(folder, "audio")):
        if filename.endswith((".wav", ".m4a", ".mp3")):
            base_name = os.path.splitext(filename)[0]
            transcript_path = os.path.join(folder, "transcripts", f"{base_name}.txt")
            if not os.path.exists(transcript_path):  # Check if transcript exists
                audio_files.append(filename)
    return audio_files


from pydub import AudioSegment
from pydub.silence import split_on_silence


# Function to intelligently split and combine chunks
def intelligent_chunking(
    audio, max_duration=15 * 60 * 1000, silence_thresh=-40, min_silence_len=500
):
    """
    Splits audio on silence and combines smaller chunks to keep each chunk under max_duration.
    :param audio: The input AudioSegment object.
    :param max_duration: Maximum duration for each chunk in milliseconds.
    :param silence_thresh: Silence threshold in dB.
    :param min_silence_len: Minimum silence length to consider as a split point in milliseconds.
    :return: A list of AudioSegment chunks.
    """
    # Step 1: Split on silence
    raw_chunks = split_on_silence(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=250
    )

    # Step 2: Combine smaller chunks into segments under the max duration
    chunks = []
    current_chunk = AudioSegment.empty()

    for chunk in raw_chunks:
        if len(current_chunk) + len(chunk) <= max_duration:
            current_chunk += chunk  # Add to the current chunk if within time limit
        else:
            # Add the current chunk to the chunks list if it exceeds max duration
            chunks.append(current_chunk)
            current_chunk = chunk  # Start a new chunk with the current segment

    # Add any remaining audio as the last chunk
    if len(current_chunk) > 0:
        chunks.append(current_chunk)

    return chunks


if __name__ == "__main__":
    # Display audio files without transcripts
    audio_files = list_audio_files_without_transcripts(temp_folder)
    print("Audio files without transcripts:")
    for i, filename in enumerate(audio_files, 1):
        print(f"{i}. {filename}")

    # User input for files to process
    choice = input(
        "\nEnter the number of the file to transcribe, or type 'all' to transcribe all: "
    ).strip()

    # Select files based on user choice
    if choice.lower() == "all":
        files_to_transcribe = audio_files
    else:
        try:
            index = int(choice) - 1
            files_to_transcribe = [audio_files[index]]
        except (ValueError, IndexError):
            print("Invalid choice. Exiting.")
            exit(1)

    # Transcribe and save results to text files
    for filename in files_to_transcribe:
        input_path = os.path.join(temp_folder, "audio", filename)
        base_name = os.path.splitext(filename)[0]
        transcript_path = os.path.join(temp_folder, "transcripts", f"{base_name}.txt")

        # Convert audio to WAV format if needed
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)  # Ensuring single-channel audio for better Whisper processing
        chunks = intelligent_chunking(audio)

        # Transcribe each chunk
        full_transcription = ""
        for idx, chunk in enumerate(chunks):
            chunk_path = f"/tmp/temp_chunk_{idx}.wav"
            chunk.export(chunk_path, format="wav")
            result = model.transcribe(chunk_path)
            full_transcription += result["text"] + "\n"

        # Write transcription to file
        with open(transcript_path, "w") as f:
            f.write(full_transcription)

    print("Transcription completed.")

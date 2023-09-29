import os
from pydub import AudioSegment

# Source folder containing .m4a files
src_folder = "audiofolder/data"

# Destination folder to save .wav files
dest_folder = "audiofolder/data"

# Create destination folder if it doesn't exist
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

# Iterate through each .m4a file in the source folder
for filename in os.listdir(src_folder):
    if filename.endswith(".m4a"):
        m4a_file_path = os.path.join(src_folder, filename)
        wav_file_path = os.path.join(dest_folder, filename.replace(".m4a", ".wav"))
        
        # Load .m4a file
        audio = AudioSegment.from_file(m4a_file_path, format="m4a")
        
        # Optional: Change audio settings
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Export to .wav format
        audio.export(wav_file_path, format="wav")
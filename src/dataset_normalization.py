from pydub import AudioSegment
from pathlib import Path
import os

database = "datatrain"
folders = os.listdir(database)
for folder in folders:
    files = os.listdir(database + "/" +folder)
    for file in files:
        audio_path = database+ "/" + folder + "/" + file
        print(audio_path)
        if audio_path.endswith('.txt'):
            continue
        filename = Path(file).stem
        audio_export = database + "/" + folder + "/" + filename + ".wav"
        song = AudioSegment.from_file(audio_path)
        song.export(audio_export, format="wav")
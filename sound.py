import time
import gtts
import openal
import os
import subprocess


def SoundFunc(xcenter, ycenter, label, distance):
    x_pos = xcenter
    y_pos = ycenter
    z_pos = 0

    # where is the object
    x_pos = (x_pos - 0.5) * 20
    y_pos = (y_pos - 0.5) * 20
    # distance_call = "at" + str(distance)
    distance_call = "".join(["at", str(distance), "meters"])

    # Texts that we want to play
    # texts = [label, distance_call, "meters"]
    texts = [label, distance_call]

    for i, text_i in enumerate(texts):
        # Make request to google to get synthesis
        tts = gtts.gTTS(text_i)
        # if i == 0 or i == 2:
        if i == 0:
            folder = "label_audio/"
            sleep_time = 1
        elif i == 1:
            folder = "distance_audio/"
            sleep_time = 2
        # Checks if the file exists
        if not os.path.isfile("".join([folder, text_i, ".wav"])):
            # Save the audio file
            tts.save(folder + text_i + ".mp3")
            # Convert mp3 to wav file
            subprocess.call(['ffmpeg', '-i', "".join([folder, text_i, ".mp3"]), "".join([folder, text_i, ".wav"])])
            # Remove mp3 file
            os.remove("".join([folder, text_i, ".mp3"]))
        # Play the file
        source = openal.oalOpen("".join([folder, text_i, ".wav"]))
        source.set_position([x_pos, y_pos, z_pos])
        source.set_looping(False)
        source.play()
        listener = openal.Listener()
        listener.set_position([0, 0, 0])
        print("Playing at: {0}".format(source.position))
        time.sleep(sleep_time)
        openal.oalQuit()

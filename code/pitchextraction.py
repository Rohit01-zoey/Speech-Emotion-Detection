import subprocess
import matplotlib.pyplot as plt

#Documentation at the following link
#https://rohitvartak.notion.site/Using-openSMILE-with-Windows-46d0e3e98aeb4df8860b54c29b53d655?pvs=4

def generate_features_with_opensmile(wav_file, output_file_csv, command_cwd, config_file, frame_rate, step_size,voicing_threshold):
    # Create a temporary output file path
    
    config_file_actual = config_file.rsplit('.', 1)[0] +"_copy."+config_file.rsplit('.', 1)[1]
    
    # Update the frame rate and step size in the config file
    with open(config_file, 'r') as file, open(config_file_actual, 'w') as output_file:
        # config_content = file.read()
        for line in file:
            if line.startswith("frameSize = "):
                line =  "frameSize = "+str(step_size) + "\n"
                output_file.write(line)
            elif line.startswith("frameStep = "):
                line = "frameStep = " + str(frame_rate) + "\n"
                output_file.write(line)
            elif line.startswith("voicingCutoff = "):
                line = "voicingCutoff = " + str(voicing_threshold) + "\n"
                output_file.write(line)
            else:
                output_file.write(line)
    
    # # Run OpenSmile command to generate features
    command = f".\SMILExtract -C {config_file_actual} -I {wav_file} -O {output_file_csv}"
    subprocess.run(command, shell=True, check=True, cwd = command_cwd)
    
    # # Read the generated features from the output file
    features = []
    with open(output_file_csv, "r") as file:
        lines = file.readlines()
        # Skip the header line
        for line in lines[1:]:
            # Split the line into values and convert to float
            values = [float(value) for value in line.strip().split(";")]
            features.append(values)
    
    # # Remove the temporary output file
    # subprocess.run(f"rm {output_file}", shell=True)
    
    return features

# Usage example

wav_file = "C:\\Users\\HP\\Downloads\\Ravdess\\audio_speech_actors_01-24\\Actor_24\\03-01-05-01-01-02-24.wav"
output_file_csv = "C:\\Users\\HP\\Downloads\\output.csv"
command_cwd = "C:\\Users\\HP\\Downloads\\opensmile\\opensmile-3.0.1-win-x64\\bin\\"
config_file = r"C:\Users\HP\Downloads\opensmile\opensmile-3.0.1-win-x64\config\demo\demo3.conf"
step_size = 0.05
frame_rate = step_size/4
voicing_threshold = 0.5

ff = generate_features_with_opensmile(wav_file, output_file_csv, command_cwd, config_file, frame_rate, step_size, voicing_threshold)

pitch = [i[-1] for i in ff]
time = [i[-3] for i in ff]
plt.figure()
plt.plot(time, pitch, '.')
plt.show()




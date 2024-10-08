import subprocess

subprocess.run(["source", "install_prereqs.sh"], shell=True, check=True)

subprocess.run(["pip", "install", "-r", "requirements.txt"], shell=True, check=True)

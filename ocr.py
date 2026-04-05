import subprocess

def capture(path="screen.png"):
    subprocess.run(
        f"xwd -root -silent -display :0 | convert xwd:- {path}",
        shell=True
    )
    return path

capture()
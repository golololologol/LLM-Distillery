import subprocess

model_path = r"F:\trained\TinyGoblin-1.1B-V3.1\TinyGoblin-1.1B-V3.1_epoch_final"

command = [
    "C:/Users/gololo/Desktop/text-generation-webui-main/installer_files/env/python.exe",
    "-m",
    "lm_eval",
    "--model", "hf",
    "--model_args", f"pretrained={model_path}",
    "--tasks", "openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq",
    "--device", "cuda:1",
    "--batch_size", "32"
]


subprocess.run(command, shell=True)

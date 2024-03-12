import subprocess
import os

path_to_convert_script = r"F:/down/koboldcpp_ggml_tools_21nov/convert_llama/convert.py"
model_path = r"F:\trained\TinyGoblin-1.1B-V3.1\TinyGoblin-1.1B-V3.1_epoch_final"

model_name = model_path.split(os.sep)[-1]
base_folder = os.path.dirname(model_path)
# make an empty gguf file to where to dump the weights
gguf_file = os.path.join(base_folder, f"{model_name}_fp16.gguf")
open(gguf_file, 'w').close()


subprocess.run(["python", path_to_convert_script, model_path, "--outtype", "f16", "--outfile", gguf_file], shell=True)
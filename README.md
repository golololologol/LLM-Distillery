# LLM-Distillery

LLM-Distillery is a pipeline for distillation of one or multiple teacher models into a student.

Main features:
* Single and Multi-Teacher distillation
* Distillation on instruct and completion text
* Offline distillation: collects the dataset, and only then trains (Yes, you can share the collected datasets)
* Windows and Linux support
* Automatic hdf5 dataset synchronization, with continued collection after force-exit
* Lots of knobs to tweak! From temperature to the device mapping strategy
* And a lot more!
  
# Installation
[See Wiki for installation instructions](https://github.com/golololologol/LLM-Distillery/wiki)

# Console UI
https://github.com/user-attachments/assets/baac01ab-a045-44ec-a752-6662e9304a60

(Full run of tinyllama 1.1B self-distillation from full fp16 model to 4bit quantized version)

# Contributions
Big thanks to [kalomaze](https://github.com/kalomaze) for help and keeping me sane while I was building this project!\
Also, thanks to [AlpinDale](https://github.com/AlpinDale) for giving access to compute during the development!
 
If you want to contribute to this project, feel free!\
Open issues when you encounter them, and make PRs when you feel like it.





import pathlib
import shutil
import os


class Paths:
    '''
    cache\n
    ├─ tensorboard_logs\n
    ├─ dataset\n
    │   └─ validation\n
    └─ student\n
        ├─ states\n
        ├─ gguf\n
        └─ trained\n
    '''

    def __init__(self, cache, clean_start: bool = False, empty_logs: bool = False):
        self.cache: str = cache
        self.logging: str = os.path.join(cache, "tensorboard_logs")
        self.dataset: str = os.path.join(cache, "dataset")
        self.dataset_validation = os.path.join(self.dataset, "validation")
        self.student_root: str = os.path.join(cache, "student")
        self.student_states: str = os.path.join(self.student_root, "states")
        self.student_gguf: str = os.path.join(self.student_root, "gguf")
        self.student_trained: str = os.path.join(self.student_root, "trained")
        self.initialize_folders(clean_start, empty_logs)
    
    def initialize_folders(self, clean_start, empty_logs=False):
        if clean_start:
            self.empty_all(empty_logs)
        os.makedirs(self.cache, exist_ok=True)
        os.makedirs(self.logging, exist_ok=True)
        os.makedirs(self.dataset, exist_ok=True)
        os.makedirs(self.dataset_validation, exist_ok=True)
        os.makedirs(self.student_root, exist_ok=True)
        os.makedirs(self.student_states, exist_ok=True)
        os.makedirs(self.student_gguf, exist_ok=True)
        os.makedirs(self.student_trained, exist_ok=True)

    def create_folder(self, existing_folder, subfolder: str):
        os.makedirs(os.path.join(existing_folder, subfolder), exist_ok=True)
        setattr(self, subfolder, os.path.join(existing_folder, subfolder))

    def empty_folder(self, folder: str):
        if not os.path.exists(folder):
            return
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                pathlib.Path(file_path).unlink()

    def empty_dataset(self):
        self.empty_folder(self.dataset)
    
    def empty_student_root(self):
        self.empty_folder(self.student_root)

    def empty_student_states(self):
        self.empty_folder(self.student_states)

    def empty_logs(self):
        self.empty_folder(self.logging)

    def empty_all(self, empty_logs=False):
        self.empty_dataset()
        self.empty_student_root()
        if empty_logs:
            self.empty_logs()

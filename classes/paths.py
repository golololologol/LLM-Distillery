import pathlib
import shutil
import os


class Paths:
    """Class to manage all paths in the project.
    
    cache\n
    ├─ dataset\n
    │   └─ validation\n
    └─ student\n
        ├─ states\n
        ├─ gguf\n
        └─ trained\n
    """

    def __init__(self, cache, clean_start: bool = False):
        self.cache: str = cache
        self.dataset: str = os.path.join(cache, "dataset")
        self.dataset_validation = os.path.join(self.dataset, "validation")
        self.student_root: str = os.path.join(cache, "student")
        self.student_states: str = os.path.join(self.student_root, "states")
        self.student_gguf: str = os.path.join(self.student_root, "gguf")
        self.student_trained: str = os.path.join(self.student_root, "trained")
        self.initialize_folders(clean_start)
    
    def initialize_folders(self, clean_start):
        if clean_start:
            self.empty_all()
        os.makedirs(self.cache, exist_ok=True)
        os.makedirs(self.dataset, exist_ok=True)
        os.makedirs(self.dataset_validation, exist_ok=True)
        os.makedirs(self.student_root, exist_ok=True)
        os.makedirs(self.student_states, exist_ok=True)
        os.makedirs(self.student_gguf, exist_ok=True)
        os.makedirs(self.student_trained, exist_ok=True)

    def create_folder(self, existing_folder, subfolder: str):
        os.makedirs(os.path.join(existing_folder, subfolder), exist_ok=True)
        setattr(self, subfolder, os.path.join(existing_folder, subfolder))

    def dataset_present(self):
        return os.path.exists(os.join(self.dataset, "distributions.hdf5"))
    
    def validation_present(self):
        return os.path.exists(os.join(self.dataset_validation, "distributions.hdf5"))

    def empty_folder(self, folder: str):
        folder_path = pathlib.Path(folder)
        if not folder_path.exists():
            return
        for file in folder_path.iterdir():
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()

    def empty_dataset(self):
        self.empty_folder(self.dataset)
    
    def empty_student_root(self):
        self.empty_folder(self.student_root)

    def empty_student_states(self):
        self.empty_folder(self.student_states)

    def empty_all(self):
        self.empty_dataset()
        self.empty_student_root()

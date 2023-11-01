import config

from lib.FileManager import FileManager


if __name__ == "__main__":
    file_manager = FileManager(root_dir=config.FILE_MANAGER_ROOT, init_dirs=True)

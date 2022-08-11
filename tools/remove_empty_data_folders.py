import argparse
import os
import shutil


def get_list_of_empty_folders(root_folder: str) -> list:
    """ Find folders with only log files in them """
    empty_folders = []
    for root, dirs, files in os.walk(root_folder):
        extensions = set()
        for file in files:
            ext = file.split(".")[1]
            extensions.add(ext)

        if len(extensions) == 1 and len(dirs) == 0 and "log" in extensions:
            empty_folders.append(root)
    return empty_folders


def remove_folders_with_choice(folders_to_remove: list):
    print("Directories with no data:")

    for _folders in folders_to_remove:
        print(_folders)

    choice = input("The above directories will be removed, continue? [y/N]")

    if choice.lower() == "y":
        for folder_to_remove in folders_to_remove:
            print(f"Removing directory {folder_to_remove}")
            try:
                shutil.rmtree(folder_to_remove)
            except PermissionError:
                print(f"Unable to remove {folder_to_remove}, it is currently in use")
    else:
        print("Not removing directories")


def dir_path(path: str) -> str:
    if os.path.isdir(path):
        return path
    else:
        raise NotADirectoryError(path)


if __name__ == "__main__":
    """
    - Walk through directories
    - Identify folders with only log files
        - loop over all files in directory
        - use a set to store all extensions
            - if the length of the set > 2, keep folder
            - else delete folder
    """
    parser = argparse.ArgumentParser(description='Delete folders with only log files in them.')
    parser.add_argument('--path', type=dir_path, help="Path to measurement folder")

    folder_path = parser.parse_args().path

    folders = get_list_of_empty_folders(folder_path)
    remove_folders_with_choice(folders)

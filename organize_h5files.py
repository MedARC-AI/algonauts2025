import os
import shutil
import argparse

def process_h5_file(h5_file_full_path, visual_dir_abs_path, suffix_to_add):
    """
    Renames an H5 file by adding a suffix and moves it to the target visual directory.
    """
    if not os.path.isfile(h5_file_full_path):
        print(f"Warning: Source file not found or is not a file: '{h5_file_full_path}', skipping.")
        return

    try:
        original_filename = os.path.basename(h5_file_full_path)
        name_part, extension = os.path.splitext(original_filename)

        if extension.lower() != ".h5":
            print(f"Warning: File '{original_filename}' is not an .h5 file, skipping.")
            return

        new_filename = f"{name_part}{suffix_to_add}{extension}"
        destination_path = os.path.join(visual_dir_abs_path, new_filename)

        print(f"Processing: Moving '{h5_file_full_path}' to '{destination_path}'")
        shutil.move(h5_file_full_path, destination_path)
    except Exception as e:
        print(f"Error processing file '{h5_file_full_path}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Organize .h5 files by renaming and moving them to a 'visual' directory."
    )
    parser.add_argument(
        "base_dir_path",
        type=str,
        help="The path to the base directory (e.g., './internvl3_8b_8bit')."
    )
    parser.add_argument(
        "suffix_to_add",
        type=str,
        help="The suffix to add to the .h5 filenames (e.g., '_features_visual')."
    )
    args = parser.parse_args()

    base_dir_abs_path = os.path.abspath(args.base_dir_path)
    suffix_to_add = args.suffix_to_add

    if not os.path.isdir(base_dir_abs_path):
        print(f"Error: Base directory '{base_dir_abs_path}' does not exist or is not a directory.")
        return

    print(f"Using base directory: {base_dir_abs_path}")
    print(f"Using suffix: \"{suffix_to_add}\"")
    print("--------------------------------------------------")

    visual_dir_abs_path = os.path.join(base_dir_abs_path, "visual")
    friends_dir_abs_path = os.path.join(base_dir_abs_path, "friends")
    # IMPORTANT: The script expects a directory named "movies".
    # If your directory is "movie10", please rename it to "movies"
    # or change the line below to:
    movies_dir_abs_path = os.path.join(base_dir_abs_path, "movie10")
    # movies_dir_abs_path = os.path.join(base_dir_abs_path, "movies")


    # 1. Ensure the target visual directory exists
    try:
        os.makedirs(visual_dir_abs_path, exist_ok=True)
        print(f"Ensuring target directory exists: {visual_dir_abs_path}")
    except Exception as e:
        print(f"Error creating visual directory '{visual_dir_abs_path}': {e}")
        return
    print("--------------------------------------------------")

    # 2. Process 'friends' directory
    print("Processing 'friends' directory...")
    if os.path.isdir(friends_dir_abs_path):
        for i in range(1, 8):  # s1 to s7
            season_dirname = f"s{i}"
            season_path_abs = os.path.join(friends_dir_abs_path, season_dirname)
            if os.path.isdir(season_path_abs):
                print(f"Scanning friends season directory: {season_path_abs}")
                for item_name in os.listdir(season_path_abs):
                    item_full_path = os.path.join(season_path_abs, item_name)
                    if os.path.isfile(item_full_path) and item_name.lower().endswith(".h5"):
                        process_h5_file(item_full_path, visual_dir_abs_path, suffix_to_add)
            else:
                print(f"Warning: Friends season directory '{season_path_abs}' not found.")
    else:
        print(f"Warning: Base 'friends' directory '{friends_dir_abs_path}' not found.")
    print("--------------------------------------------------")

    # 3. Process 'movies' directory
    print("Processing 'movies' directory (or 'movie10' if you modified the script)...")
    if os.path.isdir(movies_dir_abs_path):
        # First, process .h5 files directly under MOVIES_DIR
        print(f"Scanning files directly under: {movies_dir_abs_path}")
        for item_name in os.listdir(movies_dir_abs_path):
            item_full_path = os.path.join(movies_dir_abs_path, item_name)
            if os.path.isfile(item_full_path) and item_name.lower().endswith(".h5"):
                process_h5_file(item_full_path, visual_dir_abs_path, suffix_to_add)

        # Then, process .h5 files in first-level subdirectories of MOVIES_DIR
        print(f"Scanning first-level subdirectories under: {movies_dir_abs_path}")
        for item_name in os.listdir(movies_dir_abs_path):
            potential_subdir_path_abs = os.path.join(movies_dir_abs_path, item_name)
            if os.path.isdir(potential_subdir_path_abs):
                print(f"Scanning movie subdirectory: {potential_subdir_path_abs}")
                for sub_item_name in os.listdir(potential_subdir_path_abs):
                    sub_item_full_path = os.path.join(potential_subdir_path_abs, sub_item_name)
                    if os.path.isfile(sub_item_full_path) and sub_item_name.lower().endswith(".h5"):
                        process_h5_file(sub_item_full_path, visual_dir_abs_path, suffix_to_add)
    else:
        print(f"Warning: Base 'movies' directory '{movies_dir_abs_path}' not found (expected 'movies' or check script).")
    print("--------------------------------------------------")
    print("Script finished.")

if __name__ == "__main__":
    main()
# utils.py
import os

def make_dir(target_path) -> None:
    try:
        if not os.path.exists(target_path):
            print("Make directory %s"%(target_path))
            os.makedirs(target_path)
            #print("")
    except OSError:
        print('Error: Failed to create directory : ' +  target_path)

def check_name_available(input_name: str):
    name_to_check = input_name
    target_path = f"result/{name_to_check}"
    if not os.path.exists(target_path):
        return True
    else:
        return False
    
    
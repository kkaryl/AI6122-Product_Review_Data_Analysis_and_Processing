import pickle
import os

def is_not_empty_file_exists(filepath):
    "Check if file exist and if it is not empty"
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0

def write_to_file(filepath, listdump):
    """Write object into a file using pickle package if filepath is not empty."""
    if filepath:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
        with open(filepath, 'wb') as outfile:
            pickle.dump(listdump, outfile)
        
def load_from_file(filepath):
    """Load a file content into object using pickle package."""
    with open(filepath, 'rb') as infile:
        listdump = pickle.load(infile)
        
    return listdump
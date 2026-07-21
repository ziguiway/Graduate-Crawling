def load_conll(file_path: str) -> dict:
    
    with open(file_path, 'r') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            line = line.split("")

    
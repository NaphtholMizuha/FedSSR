from time import process_time

class Timer:
    def __init__(self, name):
        self.name = name
        self.elapsed_time = None
    def __enter__(self):
        self.start_time = process_time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = process_time() - self.start_time
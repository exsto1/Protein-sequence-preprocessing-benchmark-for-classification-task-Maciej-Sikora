from contextlib import contextmanager
import time


@contextmanager
def time_context_manager(label):
    """A context manager for timing.
       Taken from David Beazley slides on generators ('Generators: The Final Frontier')"""
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        if end - start > 60:
            if end - start > 60 * 60:
                print("{} took: {:f} h".format(label, (end - start) / 60 * 60))
            else:
                print("{} took: {:f} m".format(label, (end - start) / 60))
        else:
            print("{} took: {:f} s".format(label, (end - start)))

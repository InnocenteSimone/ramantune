import time
from functools import wraps

def measure_time(verbose=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()

            result = func(*args, **kwargs)

            elapsed = time.perf_counter() - start

            if verbose:
                if elapsed < 60:
                    time_str = f"{elapsed:.2f} seconds"
                elif elapsed < 3600:
                    time_str = f"{elapsed / 60:.2f} minutes"
                else:
                    time_str = f"{elapsed / 3600:.2f} hours"

                print(f"[TIME] {func.__name__} took {time_str}")

            return result

        return wrapper

    return decorator
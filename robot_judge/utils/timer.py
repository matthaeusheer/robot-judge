import time


class Stopwatch(object):
    """A stopwatch utility for timing execution that can be used as a regular
    object or as a context manager.
    NOTE: This should not be used an accurate benchmark of Python code, but a
    way to check how much time has elapsed between actions. And this does not
    account for changes or blips in the system clock.
    """

    def __init__(self):
        """Initialize a new `Stopwatch`, but do not start timing."""
        self.start_time = None
        self.stop_time = None

    def start(self):
        """Start timing."""
        self.start_time = time.time()

    def stop(self):
        """Stop timing."""
        self.stop_time = time.time()

    @property
    def time_elapsed(self):
        """Return the number of seconds that have elapsed since this
        `Stopwatch` started timing.
        This is used for checking how much time has elapsed while the timer is
        still running.
        """
        assert not self.stop_time, \
            "Can't check `time_elapsed` on an ended `Stopwatch`."
        return time.time() - self.start_time

    @property
    def total_run_time(self):
        """Return the number of seconds that elapsed from when this `Stopwatch`
        started to when it ended.
        """
        return self.stop_time - self.start_time

    def __enter__(self):
        """Start timing and return this `Stopwatch` instance."""
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        """Stop timing."""
        self.stop()


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        ms = (te - ts) * 1000

        if 'time_sec' in kw:
            print('Method %r finished in %2.2f sec.' % (method.__name__, ms / 1000))
        elif 'time_min' in kw:
            print('Method %r finished in %2.2f min.' % (method.__name__, ms / 1000 / 60))
        elif 'time_hour' in kw:
            print('Method %r finished in %2.2f hour.' % (method.__name__, ms / 1000 / 60 / 60))
        else:
            print('Method %r finished in %2.2f ms.' % (method.__name__, ms))

        return result
    return timed

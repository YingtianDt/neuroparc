def memorized(func):
    def wrapper(self):
        if not hasattr(self, '_cache'):
            self._cache = {}
        if func.__name__ not in self._cache:
            self._cache[func.__name__] = func(self)
        return self._cache[func.__name__]
    return wrapper
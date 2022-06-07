from typing import Optional


class TTLContext(object):
    def __init__(self, name: str, lifetime: Optional[int] = 3):
        self.name = name
        self.lifetime = lifetime  # if None, then the contexts cannot die
        self.remaining = lifetime
        self.lived = 0

    def live(self):
        self.lived += 1
        if self.lifetime is not None:
            self.remaining = max(-1, self.remaining - 1)
    @property
    def is_dead(self) -> bool:
        if self.lifetime is None:
            return False
        return self.remaining < 0

    def __repr__(self):
        return '("{}": {})'.format(self.name, self.remaining)


if __name__ == '__main__':
    c = TTLContext('my-context', lifetime=1)
    while not c.is_dead:
        print(c)
        print(c.lifetime)
        print(c.lived)
        print(c.remaining)
        c.live()

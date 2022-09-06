class GlobalInfo(object):
    step = 0
    mode = ''

    @classmethod
    def count(cls):
        cls.step += 1

    @classmethod
    def set_mode(cls, mode):
        cls.mode = mode

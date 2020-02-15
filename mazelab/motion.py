from collections import namedtuple


BaseMotion = namedtuple('BaseMotion', ['up', 'down', 'left', 'right'],
                         defaults = [[-1,0], [1,0], [0,-1], [0,1]])

from collections import OrderedDict
class Problem():
    def __init__(self):
        nparam = 10
        space = OrderedDict()
        #problem specific parameters
        for i in range(nparam):
            space['p%d'%(i+1)] = (-2,2)
        self.space = space
        self.params = self.space.keys()
        self.starting_point = [-2] * nparam

if __name__ == '__main__':
    instance = Problem()
    print(instance.space)
    print(instance.params)



class Politique:
    def __init__(self, nb_action):
        pass

    def get_choice(self, ctx):
        pass

    def update(self, ctx, a, r):
        pass

class PolitiqueConstant:
    def __init__(self, nb_action):
        pass

    def get_choice(self, ctx):
        pass

    def update(self, ctx, a, r):
        pass

def evaluation(file, politique):
    
    with open(file, 'r') as f:
        for line in f:
            line = f.readline()
            num_article = line.split(':')[0]
            dimensions = line.split(':')[1].split(';')
            ctx = list(map(lambda x: float(x), line.split(':')[2].split(';')))

            r = 0
            a = politique.get_choice(ctx)
            politique.update(ctx, a, r)


if __name__ == "__main__":
    pass
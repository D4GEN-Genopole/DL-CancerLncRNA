from models.random.random import RandomModel
from utils.evaluate import ExpressionsEvaluator


class MainCLI(object):
    def __init__(self):
        pass

    def main(self):
        model = RandomModel()
        evaluator = ExpressionsEvaluator(model)
        scores = evaluator.evaluate()

if __name__ == '__main__':
    MainCLI().main()
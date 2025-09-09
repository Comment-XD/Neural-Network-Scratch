class Trainer:
    def __init__(self, model, X, y, loss):
        self.model = model
        self.X = X
        self.y = y
        self.loss = loss
        self.loss_chart = []

    def run(self, alpha:float=0e-4, verbose:bool=False, epochs:int=100):
        self.epochs = epochs

        for i in range(epochs):

            output = self.model(self.X)
            loss = self.loss(self.y, output)
            self.loss_chart.append(loss)

            if verbose:
                if i % (epochs // 10) == 0:
                    print(f"Epoch {i+1}/{epochs}")
                    print(f"| {self.loss.__class__.__name__} -> {loss: .4f} |")
                    print()

            gradient = self.loss.backward()
            self.model.backward(gradient, alpha)
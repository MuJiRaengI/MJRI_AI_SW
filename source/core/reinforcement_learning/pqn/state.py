class CustomTrainState:
    def __init__(self, model, optimizer, timesteps=0, n_updates=0, grad_steps=0):
        self.model = model
        self.optimizer = optimizer
        self.timesteps = timesteps
        self.n_updates = n_updates
        self.grad_steps = grad_steps

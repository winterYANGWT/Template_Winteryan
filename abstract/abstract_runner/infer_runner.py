from abc import ABC


class InferRunner(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.load_config()
        self.load_transforms()
        self.load_models()

    @classmethod
    def load_config(self):
        '''
        Load config from config package.
        '''
        msg = 'load_config should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def load_transforms(self):
        '''
        Load transform(s) from transform package.
        '''
        msg = 'load_transforms should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def load_models(self):
        '''
        Load model(s) from model package.
        '''
        msg = 'load_models should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def get_data(self):
        '''
        Get data from user's input.
        '''
        msg = 'get_data should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def forward(self, input_data):
        '''
        Forward inference.
        '''
        msg = 'forward should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def save_results(self, input_data, output_data):
        '''
        Save results.
        '''
        msg = 'save_results should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def infer(self):
        '''
        Inference phase.
        '''
        msg = 'infer should be implemented by subclass.'
        raise NotImplementedError(msg)

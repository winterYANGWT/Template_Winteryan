from abc import ABC, abstractclassmethod


class AbstractStep(ABC):
    '''
    These class method should be implemented by subclass:
        `load_config`,
        `load_data`,
        `load_models`,
        `load_optimizers`,
        `load_criterions`,
        `load_meters`,
        `forward`,
        `backward`
    '''
    def __init__(self) -> None:
        super().__init__()
        self.load_config()
        self.load_datasets()
        self.load_models()
        self.load_optimizers()
        self.load_criterions()
        self.load_meters()

    @abstractclassmethod
    def load_config(self):
        '''
        Load config.
        '''
        msg = 'load_config should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def load_datasets(self):
        '''
        Load datasets. All datasets will be contained in member variable self.datasets(dict) and generate the corresponding member variable self.data_loaders(dict).
        '''
        msg = 'load_datasets should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def load_models(self):
        '''
        Load models.
        '''
        msg = 'load_model should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def load_optimizers(self):
        '''
        Load optimizers.
        '''
        msg = 'load_optimizers should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def load_criterions(self):
        '''
        load criterions.
        '''
        msg = 'load_criterions should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def load_meters(self):
        '''
        load meters.
        '''
        msg = 'load_meters should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def forward(self, input_data):
        '''
        forward inference.
        '''
        msg = 'forward should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def backward(self, input_data, output_data):
        '''
        backward propagation.
        '''
        msg = 'backward should be implemented by subclass.'
        raise NotImplementedError(msg)

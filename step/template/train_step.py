from abc import ABC, abstractclassmethod, abstractmethod


class TrainStep(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.load_config()
        self.load_datasets()
        self.load_models()
        self.load_optimizers()
        self.load_criterions()
        self.load_meters()

    @abstractmethod
    def load_config(self):
        '''
        Load config from config package.
        '''
        msg = 'load_config should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def load_datasets(self):
        '''
        Load dataset(s) from dataset package.
        '''
        msg = 'load_datasets should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def load_models(self):
        '''
        Load model(s) from model package.
        '''
        msg = 'load_models should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def load_optimizers(self):
        '''
        Load optimizer(s) for model(s).
        '''
        msg = 'load_optimizers should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def load_criterions(self):
        '''
        Load criterion(s) from loss package.
        '''
        msg = 'load_criterions should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def load_meters(self):
        '''
        Load meter(s) from meter package.
        '''
        msg = 'load_meters should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def forward(self, input_data):
        '''
        Forward inference.
        '''
        msg = 'forward should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def backward(self, input_data, output_data):
        '''
        Backward propagation.
        '''
        msg = 'backward should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def train(self, epoch):
        '''
        Train phase.
        '''
        msg = 'train should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def val(self, epoch):
        '''
        Validate phase.
        '''
        msg = 'val should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def save_models(self, save_dir):
        '''
        Save model(s).
        '''
        msg = 'save_models should be implemented by subclass.'
        raise NotImplementedError(msg)

    @abstractclassmethod
    def update_meters(input_data, output_data):
        '''
        Update meter(s).
        '''
        msg = 'update_meters should be implemented by subclass.'
        raise NotImplementedError(msg)

from abc import ABC


class TrainStep(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.load_config()
        self.load_datasets()
        self.load_models()
        self.load_optimizers()
        self.load_criterions()
        self.load_meters()

    @classmethod
    def load_config(self):
        '''
        Load config from config package.
        '''
        msg = 'load_config should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def load_datasets(self):
        '''
        Load dataset(s) from dataset package.
        '''
        msg = 'load_datasets should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def load_models(self):
        '''
        Load model(s) from model package.
        '''
        msg = 'load_models should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def load_optimizers(self):
        '''
        Load optimizer(s) for model(s).
        '''
        msg = 'load_optimizers should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def load_criterions(self):
        '''
        Load criterion(s) from loss package.
        '''
        msg = 'load_criterions should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def load_meters(self):
        '''
        Load meter(s) from meter package.
        '''
        msg = 'load_meters should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def forward(self, input_data):
        '''
        Forward inference.
        '''
        msg = 'forward should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def backward(self, input_data, output_data):
        '''
        Backward propagation.
        '''
        msg = 'backward should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def train(self, epoch):
        '''
        Train phase.
        '''
        msg = 'train should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def val(self, epoch):
        '''
        Validate phase.
        '''
        msg = 'val should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def save_models(self, save_dir):
        '''
        Save model(s).
        '''
        msg = 'save_models should be implemented by subclass.'
        raise NotImplementedError(msg)

    @classmethod
    def update_meters(input_data, output_data):
        '''
        Update meter(s).
        '''
        msg = 'update_meters should be implemented by subclass.'
        raise NotImplementedError(msg)


class InferStep(ABC):
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


class TestStep(ABC):
    def __init__(self) -> None:
        super().__init__()

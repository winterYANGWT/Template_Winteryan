__all__ = ['initialize_meters', 'initialize_optimizers']


def initialize_meters(meters):
    for mode in meters.keys():
        for meter_key in meters[mode].keys():
            meters[mode][meter_key].reset()


def initialize_optimizers(optimizers):
    for optimizer_name in optimizers.keys():
        if optimizers[optimizer_name] != None:
            optimizers[optimizer_name].zero_grad()

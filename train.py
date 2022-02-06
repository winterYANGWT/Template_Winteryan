from step import TrainStep
import torch

step = TrainStep()
torch.set_num_threads(step.cfg.NUM_THREADS)
torch.manual_seed(step.cfg.SEED)

for epoch in range(step.cfg.START_EPOCH, step.cfg.EPOCHES):
    step.train(epoch)
    step.save_models(str(epoch))
    step.val(epoch)
    step.save_models('best')

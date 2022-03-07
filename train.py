from step import TrainStep
import torch
import os.path as path

step = TrainStep()
torch.manual_seed(step.cfg.SEED)

for epoch in range(step.cfg.START_EPOCH, step.cfg.EPOCHES + 1):
    step.train(epoch)
    step.save_models(path.join(step.cfg.MODEL_DIR, str(epoch)))
    step.val(epoch)
    step.save_models(path.join(step.cfg.MODEL_DIR, 'best'))

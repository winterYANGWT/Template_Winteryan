from runner import TrainRunner
import torch
import os.path as path
from tqdm import tqdm

runner = TrainRunner()
cfg = runner.cfg
torch.manual_seed(cfg.SEED)

with tqdm(range(cfg.START_STEP + 1, cfg.STEPS + 1),
          desc='train',
          initial=cfg.START_STEP,
          total=cfg.STEPS,
          dynamic_ncols=True,
          ascii=True,
          colour='green') as t:
    initial_flag = True

    for step in t:
        runner.train(step, initial_flag)
        t.set_postfix_str(runner.meters['train'])
        initial_flag = False

        if step % cfg.VAL_EVERY == 0:
            runner.save_models(path.join(cfg.MODEL_DIR, str(step)))
            runner.val(step)
            runner.save_models(path.join(cfg.MODEL_DIR, 'best'))
            initial_flag = True

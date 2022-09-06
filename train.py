from runner import TrainRunner
import torch
from tqdm import tqdm
from tool import GlobalInfo

runner = TrainRunner()
cfg = runner.cfg
print(cfg)
torch.manual_seed(cfg.train.seed)
GlobalInfo.step = cfg.train.start_step

with tqdm(range(cfg.train.start_step + 1, cfg.train.steps + 1),
          desc='train',
          initial=cfg.train.start_step,
          total=cfg.train.steps,
          dynamic_ncols=True,
          ascii=True,
          colour='green') as t:
    initial_flag = True

    for step in t:
        GlobalInfo.count()
        GlobalInfo.set_mode('train')
        runner.train(step, initial_flag)
        t.set_postfix_str(runner.meter_manager)
        initial_flag = False

        if step % cfg.train.val_every == 0:
            GlobalInfo.set_mode('eval')
            runner.save_models(str(step))
            runner.eval(step)
            runner.save_models('best')
            initial_flag = True

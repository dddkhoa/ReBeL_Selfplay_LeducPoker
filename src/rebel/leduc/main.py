import torch
from model import Net
import time
import os
import logging
from tqdm import tqdm

from Buffer import ReplayBuffer, DataLoop
from Game import Game
from LimitSolver import SolverParam, CFRParam
from LeducNet import TrainDataNet


def huber_loss(diff: torch.Tensor, delta=1):
    diff_abs = diff.abs()
    return (diff_abs > delta).float() * (2 * delta * diff_abs - delta ** 2) + (diff_abs <= delta).float() * diff.pow(2)


if __name__ == '__main__':
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter('%(asctime)s - %(message)s')

    input_dim, output_dim = 1 + 1 + 1 + 6 + 2 * 6, 6
    hidden_dims = [64, 64, 128]
    model = Net(input_dim, hidden_dims, output_dim)
    script_model = torch.jit.script(model)

    actor_device = 'cpu'
    act_model = Net(input_dim, hidden_dims, output_dim)
    act_model.to(actor_device)
    act_model = torch.jit.script(act_model)
    act_model.eval()

    replay_buffer = ReplayBuffer(2 ** 20, 2022)

    game = Game()

    cfr_param = CFRParam()
    cfr_param.discount = True
    cfr_param.alpha = 1.5
    cfr_param.beta = 0
    cfr_param.gamma = 2

    param = SolverParam()

    net = TrainDataNet(replay_buffer)
    loop = DataLoop(game, cfr_param, param, net, 2022).loop()

    train_device = torch.device('cpu')

    model.train()
    model.to(train_device)
    opt = torch.optim.Adam(model.parameters())
    max_epochs = 10
    purging_epochs = set([20])
    epoch_batch = 1
    batch_size = 1
    epoch_size = epoch_batch * batch_size
    decrease_lr_every = 1
    network_sync_epochs = 1
    train_gen_ratio = 4
    grad_clip = 5.0
    ckpt_every = 5

    ckpt_dir = '../model/test'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    for epoch in tqdm(range(max_epochs)):
        print(f"Running epoch: {epoch}")
        if purging_epochs is not None and epoch in purging_epochs:
            size = len(replay_buffer)
            replay_buffer.pop_until(size // 2)
            print("purging buffer size at epoch %d : %d-->%d" % (epoch, size, len(replay_buffer)))
        if epoch != 0 and epoch % decrease_lr_every == 0:
            for param_group in opt.param_groups:
                param_group['lr'] /= 2
            print("decrease lr at epoch %d " % epoch)
        if train_gen_ratio is not None:
            while True:
                num_add = len(replay_buffer)
                if num_add * train_gen_ratio >= epoch * epoch_size:
                    break
                print('%d*%d < %d*%d' % (num_add, train_gen_ratio, epoch, epoch_size))
                time.sleep(60)

        for batch in tqdm(range(epoch_batch)):
            data = replay_buffer.pick_experiences(batch_size)
            feature, target = data[0].feature.to(train_device), data[0].target.to(train_device)
            loss = (huber_loss(model(feature) - target)).mean()
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            print('Epoch: %d\t Batch: %d\t Loss: %f' % (epoch, batch, loss))

        if (epoch + 1) % network_sync_epochs == 0:
            print('update model at epoch %d ' % epoch)
        if (epoch + 1) % ckpt_every == 0:
            ckpt_path = '%s/epoch_%d' % (ckpt_dir, epoch)
            torch.save(model, ckpt_path + '.ckpt')
            torch.jit.save(torch.jit.script(model), ckpt_path + '.torchscript')
            print('save ' + ckpt_path + '.ckpt')
            print('save ' + ckpt_path + '.torchscript')

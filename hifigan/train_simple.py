"""
Simplified `train.py`. (`train.py` is preserved for diff toward HiFi-GAN official)
"""

import itertools
import os
import time
import argparse
import json

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp.grad_scaler import GradScaler
from fastprogress import master_bar, progress_bar

from .utils import AttrDict, build_env
from .meldataset import MelDataset, mel_spectrogram, get_dataset_filelist, LogMelSpectrogram
from .models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, discriminator_loss                                                                       # [Diff] relative import
from .utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint


torch.backends.cudnn.benchmark = True


def train(a, h):

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:0')

    # Models - G/MPD/MSD/melnizer
    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    alt_melspec = LogMelSpectrogram(h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax).to(device)

    os.makedirs(a.checkpoint_path, exist_ok=True)
    print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
        print(f"Restored checkpoint from {cp_g} and {cp_do}") # [Diff] Detail report

    # Optim/Sched
    optim_g = torch.optim.AdamW(generator.parameters(),                              h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    # FP16
    if a.fp16:                  # [Diff] AMP
        scaler_g = GradScaler() # [Diff] AMP
        scaler_d = GradScaler() # [Diff] AMP

    # Data
    training_filelist, validation_filelist = get_dataset_filelist(a)
    trainset = MelDataset(training_filelist,   h.segment_size, h.n_fft, h.num_mels, h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax,               n_cache_reuse=0,
                          shuffle=True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, audio_root_path=a.audio_root_path, feat_root_path=a.feature_root_path, use_alt_melcalc=True)
    validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels, h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                                        fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, audio_root_path=a.audio_root_path, feat_root_path=a.feature_root_path, use_alt_melcalc=True)
    train_loader      = DataLoader(trainset, num_workers=h.num_workers, shuffle=False, sampler=None, batch_size=h.batch_size, pin_memory=True, drop_last=True)
    validation_loader = DataLoader(validset, num_workers=1,             shuffle=False, sampler=None, batch_size=1,            pin_memory=True, drop_last=True)

    sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    
    mb = master_bar(range(max(0, last_epoch), a.training_epochs))
    # mb = range(max(0, last_epoch), a.training_epochs)
    for epoch in mb:
        mb.write("Epoch: {}".format(epoch+1))
        for i, batch in progress_bar(enumerate(train_loader), total=len(train_loader), parent=mb):
        # for i, batch in              enumerate(train_loader):

            # Data
            unit_series, y, _, y_mel = batch
            unit_series = unit_series.to(device, non_blocking=True)
            y           =           y.to(device, non_blocking=True)
            y_mel       =       y_mel.to(device, non_blocking=True)
            y = y.unsqueeze(1)

            # Common_Forward - unit-to-wave & wave_pred melnize
            with torch.cuda.amp.autocast(enabled=a.fp16):
                y_g_hat = generator(unit_series)
                y_g_hat_mel = alt_melspec(y_g_hat.squeeze(1))

            # D_Forward - MPD & MSD
            optim_d.zero_grad()
            with torch.cuda.amp.autocast(enabled=a.fp16):
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            # D_Loss - adv_d_mpd & adv_d_msd
                loss_disc_f = discriminator_loss(y_df_hat_r, y_df_hat_g)[0]
                loss_disc_s = discriminator_loss(y_ds_hat_r, y_ds_hat_g)[0]
                loss_disc_all = loss_disc_s + loss_disc_f
            # D_Backward/Optim
            if a.fp16:
                scaler_d.scale(loss_disc_all).backward()
                scaler_d.step(optim_d)
                scaler_d.update()
            else:
                loss_disc_all.backward()
                optim_d.step()

            # Generator
            optim_g.zero_grad()
            # G_Loss - Adv_g / FM / Mel
            with torch.cuda.amp.autocast(enabled=a.fp16):
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
                _, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                _, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f = generator_loss(y_df_hat_g)[0]
                loss_gen_s = generator_loss(y_ds_hat_g)[0]
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            # G_Backward/Optim
            if a.fp16:
                scaler_g.scale(loss_gen_all).backward()
                scaler_g.step(optim_g)
                scaler_g.update()
            else:
                loss_gen_all.backward()
                optim_g.step()

            # STDOUT logging
            if steps % a.stdout_interval == 0:
                with torch.no_grad():
                    mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                mb.write('Steps : {:,d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, peak mem: {:5.2f}GB'. \
                        format(steps, loss_gen_all, mel_error, torch.cuda.max_memory_allocated()/1e9))
                mb.child.comment = "Steps : {:,d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}". \
                        format(steps, loss_gen_all, mel_error)

            # checkpointing
            if steps % a.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}.pt".format(a.checkpoint_path, steps)  # [Diff] path
                save_checkpoint(checkpoint_path,
                                {'generator': generator.state_dict()})
                checkpoint_path = "{}/do_{:08d}.pt".format(a.checkpoint_path, steps) # [Diff] path
                save_checkpoint(checkpoint_path, 
                                {'mpd': mpd.state_dict(), 'msd': msd.state_dict(),
                                    'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps, 'epoch': epoch})

            # Tensorboard summary logging
            if steps % a.summary_interval == 0:
                sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                sw.add_scalar("training/mel_spec_error", mel_error, steps)
                sw.add_scalar("training/disc_loss_total", loss_disc_all, steps) # [Diff] Detail report

            # Validation
            if steps % a.validation_interval == 0:  # and steps != 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0
                with torch.no_grad():
                    for j, batch in progress_bar(enumerate(validation_loader), total=len(validation_loader), parent=mb): # [Diff] fastprogress
                        x, y, _, y_mel = batch
                        y_g_hat = generator(x.to(device))
                        y_mel = y_mel.to(device, non_blocking=True)
                        y_g_hat_mel = alt_melspec(y_g_hat.squeeze(1))
                        if y_g_hat_mel.shape[-1] != y_mel.shape[-1]:
                            # pad it
                            n_pad = h.hop_size
                            y_g_hat = F.pad(y_g_hat, (n_pad//2, n_pad - n_pad//2))
                            y_g_hat_mel = alt_melspec(y_g_hat.squeeze(1))
                        val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                        if j <= 4:
                            if steps == 0:
                                sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                            sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                            y_hat_spec = alt_melspec(y_g_hat.squeeze(1))
                            sw.add_figure('generated/y_hat_spec_{}'.format(j), plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                    val_err = val_err_tot / (j+1)
                    sw.add_scalar("validation/mel_spec_error", val_err, steps)
                    mb.write(f"validation run complete at {steps:,d} steps. validation mel spec error: {val_err:5.4f}") # [Diff] Detail report

                generator.train()
                sw.add_scalar("memory/max_allocated_gb", torch.cuda.max_memory_allocated()/1e9, steps) # [Diff] Detail report
                sw.add_scalar("memory/max_reserved_gb", torch.cuda.max_memory_reserved()/1e9, steps)   # [Diff] Detail report
                torch.cuda.reset_peak_memory_stats()                                                   # [Diff] Detail report
                torch.cuda.reset_accumulated_memory_stats()                                            # [Diff] Detail report

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        

def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--audio_root_path', required=True)   # [Diff] `input_wavs_dir` -> `audio_root_path`
    parser.add_argument('--feature_root_path', required=True) # [Diff] `input_mels_dir` -> `feature_root_path`
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=1500, type=int)     # [Diff] 3100 -> 1500
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=25, type=int)      # [Diff]  100 ->   25
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', action='store_true') # [Diff] Argparse type
    parser.add_argument('--fp16', default=False, type=bool)

    a = parser.parse_args()
    print(a)

    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    # Save config under checkpoint dir
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = 1

    assert h.num_gpus == 1, "Currently support only h.num_gpus==1"

    train(a, h)


if __name__ == '__main__':
    main()

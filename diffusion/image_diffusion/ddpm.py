'''
Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC
 
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import torch
import numpy as np
from flowers import get_flowers
from unet import UNet2D
import cv2
from tqdm import tqdm
from utility import get_diffusion_parameters, cosine_lr_scheduler
from generate import generate_sample
import matplotlib.pyplot as plt
from config import Config
import os
import pickle


cfg = Config()
batch_size = cfg.batch_size
epochs = cfg.epochs
ema = cfg.use_ema

if not os.path.exists(cfg.output):
    os.makedirs(cfg.output)

# pre-calculate coefficients as a function of time t
betas, alphas, alpha_hats = get_diffusion_parameters(cfg.T)

# get dataset
ds = get_flowers(cfg.resolution)

# set up model
device = torch.device("cuda")
model = UNet2D(in_channels=3, in_side_dim=cfg.resolution, filters=cfg.filters, attn=cfg.attn, T=cfg.T).to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
total_steps = (len(ds)/batch_size)*epochs
lr_sched = cosine_lr_scheduler(opt, total_steps, warmup_steps=500, final=0.001)

if ema:
    ema_model = torch.optim.swa_utils.AveragedModel(model, 
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))

if cfg.mixed_precision:
    scaler = torch.cuda.amp.GradScaler()

params = sum(p.numel() for p in model.parameters())
print("---")
print("Model has", f'{params:,}', "parameters.")
print("---")

# train
dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
best_loss = 100000.0
loss_history = []

for epoch in range(epochs):
    print("EPOCH", epoch+1)

    epoch_losses = []
    for batch in tqdm(iter(dataloader)):
        opt.zero_grad()
        x = batch["image"].to(torch.float32).to(device)
        B = x.shape[0]
        side = x.shape[-1]

        # also sample noise, times, and alpha_hats
        e = torch.randn(*x.shape).to(torch.float32).to(device)
        t = np.random.randint(0, cfg.T, size=(B,))
        ahat = alpha_hats[t]
        t = torch.as_tensor(t).to(torch.long).to(device)
        ahat = torch.as_tensor(ahat).to(torch.float32).to(device)
        ahat = ahat.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,3,side,side)

        noised_batch = torch.sqrt(ahat)*x + torch.sqrt(1.0 - ahat)*e

        if cfg.mixed_precision:
            with torch.amp.autocast(device_type="cuda", dtype=cfg.mixed_type):
                predicted_noise = model(noised_batch, t)
                loss = torch.nn.functional.mse_loss(predicted_noise, e)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        else:
            predicted_noise = model(noised_batch, t)
            loss = torch.nn.functional.mse_loss(predicted_noise, e)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        epoch_losses.append(loss.item())
        
        if ema:
            ema_model.update_parameters(model)

        lr_sched.step()


    final_loss = np.mean(epoch_losses[-20:])
    print("Avg loss:", np.mean(epoch_losses), "Final loss:", np.mean(epoch_losses[-10:]))
    loss_history.append(np.mean(epoch_losses))

    # save a loss curve
    if len(loss_history) > 1:
        plt.clf()
        plt.plot(loss_history)
        plt.ylim(0.0, 0.1)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(cfg.output+"/loss.png")

        pickle.dump( loss_history, open( cfg.output+"losses.p", "wb" ) )

    # save model
    if final_loss < best_loss:
        torch.save(model.state_dict(), cfg.output+"/raw_ddpm_model.pt")
        if ema:
            torch.save(ema_model.state_dict(), cfg.output+"/ema_ddpm_model.pt")
        best_loss = final_loss
        print("New best model saved.")

    # periodically save a sample
    if (epoch+1) % 20 == 0:
        print("Generating samples.")
        image = generate_sample(model if (not ema) else ema_model, cfg, device, "square")
        cv2.imwrite(cfg.output+'/generations_epoch_'+str(epoch+1)+'.png', image[:,:,::-1])
# import inspect
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
#
# def train_cnn_kernel(model, trainingdata,
#                      dt, dx, mu, epochs=5000, learning_rate=1e-3,
#                      added_constraints = False,
#                      l1=0.0, l2=0.0, l3=0.0,   #Soft constraints on consistency, etc
#                      multistep=False, plot=False,
#                      scheduler=True, breaktol=1e-7,
#                      conv_stats=False, compile_model=False,
#                      multicase=False, batch_size=64, shuffle=True):
#     """
#     Trainer (enhanced) - retains original Trainer1 behaviour for single time-series input,
#     and supports multicase training when trainingdata is (inputs, targets, bcs) or multicase=True.
#
#     - trainingdata: either
#         * numpy array shape (Ntime, nx)  (original behaviour), or
#         * tuple/list: (combined_inputs, combined_targets, combined_bcs) where
#             combined_inputs: (Nsamples, nx)
#             combined_targets: (Nsamples, nx)
#             combined_bcs: (Nsamples, 2)
#       Set multicase=True to force DataLoader mode.
#     - multistep: keep as original (but note: multicase mode best used with multistep=1)
#     """
#     CFL = mu * dt / dx**2
#
#     if added_constraints and hasattr(added_constraints, "__len__"):
#         l1 = added_constraints[0]
#         l2 = added_constraints[1]
#         l3 = added_constraints[2]
#
#     # device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#
#     # Compile optional
#     if compile_model and hasattr(torch, 'compile'):
#         try:
#             model = torch.compile(model)
#             print("Model compiled successfully")
#         except Exception as e:
#             print(f"Model compilation failed: {e}")
#
#     # --- Decide training mode: original single series or multicase dataloader ---
#     use_dataloader = False
#     if multicase:
#         use_dataloader = True
#     else:
#         # autodetect tuple/list form
#         if isinstance(trainingdata, (list, tuple)) and len(trainingdata) >= 2:
#             use_dataloader = True
#
#     # Scheduler options canonicalization
#     scheduler_type = None
#     scheduler_obj = None
#
#     # --- Prepare DataLoader path (multicase) ---
#     if use_dataloader:
#         # Expect (inputs, targets, bcs) in numpy arrays (already shuffled by user)
#         if isinstance(trainingdata, (list, tuple)):
#             combined_inputs, combined_targets = trainingdata[0], trainingdata[1]
#             combined_bcs = trainingdata[2] if len(trainingdata) > 2 else None
#         else:
#             raise ValueError("For multicase/dataloader mode, provide trainingdata as (inputs,targets,bcs).")
#
#         inputs = torch.tensor(combined_inputs, dtype=torch.float32)
#         targets = torch.tensor(combined_targets, dtype=torch.float32)
#         if combined_bcs is not None:
#             bcs = torch.tensor(combined_bcs, dtype=torch.float32)
#         else:
#             bcs = None
#
#         # Add channel dim for CNN (N, 1, nx)
#         inputs = inputs.unsqueeze(1)
#         targets = targets.unsqueeze(1)
#
#         # Move to device later inside batch loop to allow DataLoader CPU->GPU transfers
#         dataset = torch.utils.data.TensorDataset(inputs, targets, bcs) if bcs is not None else torch.utils.data.TensorDataset(inputs, targets)
#         loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#         # Setup scheduler object properly
#         if scheduler == "ReduceLROnPlateau":
#             scheduler_obj = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                                  factor=0.7, patience=300, verbose=True)
#             scheduler_type = "ReduceLROnPlateau"
#         elif scheduler == "OneCycleLR":
#             # OneCycleLR must know steps_per_epoch (batches per epoch)
#             steps_per_epoch = max(1, len(loader))
#             scheduler_obj = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
#                                                          epochs=epochs, steps_per_epoch=steps_per_epoch)
#             scheduler_type = "OneCycleLR"
#         elif scheduler == "StepLR":
#             scheduler_obj = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
#             scheduler_type = "StepLR"
#         elif scheduler is True:
#             # default to StepLR for compatibility
#             scheduler_obj = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
#             scheduler_type = "StepLR"
#         else:
#             scheduler_obj = None
#
#         criterion = nn.MSELoss()
#
#         # plotting / stats
#         if plot:
#             fig, ax = plt.subplots(1, 2, figsize=(18, 6))
#             plt.ion(); plt.show()
#
#         if conv_stats:
#             Nsize = 1
#             Nsize += model.N
#             if added_constraints:
#                 try:
#                     Nsize += len(added_constraints)
#                 except Exception:
#                     pass
#             conv_hist = np.zeros([epochs, Nsize])
#
#         # Training loop (mini-batches)
#         for epoch in range(epochs):
#             model.train()
#             total_loss = 0.0
#             num_batches = 0
#
#             for batch in loader:
#                 # batch is either (inputs, targets) or (inputs, targets, bcs)
#                 if bcs is not None:
#                     batch_x, batch_y, batch_bc = batch
#                 else:
#                     batch_x, batch_y = batch
#                     batch_bc = None
#
#                 batch_x = batch_x.to(device)
#                 batch_y = batch_y.to(device)
#                 if batch_bc is not None:
#                     batch_bc = batch_bc.to(device)
#
#                 optimizer.zero_grad(set_to_none=True)
#
#                 # call model with/without bcs depending on its signature
#                 try:
#                     # model may optionally accept bcs
#                     sig = inspect.signature(model.forward)
#                     if 'bcs' in sig.parameters or 'bc' in sig.parameters or len(sig.parameters) >= 2:
#                         # try call with bc if we have it
#                         if batch_bc is not None:
#                             predictions = model(batch_x, batch_bc)
#                         else:
#                             predictions = model(batch_x)
#                     else:
#                         predictions = model(batch_x)
#                 except Exception:
#                     # fallback: try both
#                     try:
#                         predictions = model(batch_x, batch_bc)
#                     except Exception:
#                         predictions = model(batch_x)
#
#                 data_loss = criterion(predictions, batch_y)
#
#                 cl1, cl2, cl3 = 0.0, 0.0, 0.0
#                 if hasattr(model, 'constraint_loss'):
#                     try:
#                         cl1, cl2, cl3 = model.constraint_loss(CFL)
#                     except Exception:
#                         cl1, cl2, cl3 = 0.0, 0.0, 0.0
#
#                 if added_constraints:
#                     loss = data_loss + l1 * cl1 + l2 * cl2 + l3 * cl3
#                 else:
#                     loss = data_loss
#
#                 loss.backward()
#                 optimizer.step()
#                 if scheduler_obj is not None and scheduler_type == "OneCycleLR":
#                     # OneCycleLR expects per-batch step
#                     scheduler_obj.step()
#
#                 total_loss += loss.item()
#                 num_batches += 1
#
#             # end batch loop
#             avg_loss = total_loss / max(1, num_batches)
#
#             # step epoch-level schedulers
#             if scheduler_obj is not None and scheduler_type == "ReduceLROnPlateau":
#                 scheduler_obj.step(avg_loss)
#             elif scheduler_obj is not None and scheduler_type == "StepLR":
#                 scheduler_obj.step()
#
#             # collect conv stats at epoch end
#             if conv_stats:
#                 w = model.conv.weight[:].cpu().detach().numpy()[0][0]
#                 conv_hist[epoch, 0] = avg_loss
#                 conv_hist[epoch, 1:model.N+1] = w[:model.N]
#                 if added_constraints:
#                     conv_hist[epoch, model.N+1:model.N+1+3] = [float(cl1), float(cl2), float(cl3)]
#
#             # plotting if needed (keep behaviour similar to original)
#             if plot:
#                 print("Epoch ", epoch , "Loss:", avg_loss,
#                       "weights:", model.conv.weight[:].cpu().detach().numpy())
#                 # NOTE: plotting of time-series grids doesn't make sense for shuffled minibatches;
#                 # so we skip that here to avoid indexing errors.
#
#             if avg_loss < breaktol:
#                 break
#
#             # periodic logging (similar to original)
#             if (epoch + 1) % 100 == 0:
#                 w = model.conv.weight[:].cpu().detach().numpy()[0][0]
#                 j_values = np.arange(-(model.N // 2), (model.N // 2) + 1)
#                 sum_bj = w.sum()
#                 sum_bj_j = (w * j_values).sum()
#                 sum_bj_j2 = (w * j_values**2).sum()
#                 weights = w.copy() / CFL
#                 mid = int(w.shape[0] / 2.)
#                 weights[mid] = (w[mid] - 1.) / CFL
#                 print("Epoch ", epoch , "/", epochs,
#                       "Loss:", "{:.3e}".format(avg_loss),
#                       "weights:", np.round(weights,8),
#                       "sum w",  np.round(w.sum(),8),
#                       "sum wj",  np.round((w * j_values).sum(),8),
#                       "2CFL-sum wj^2",  2.*CFL-np.round((w * j_values**2).sum(),8))
#
#         # end epoch loop
#
#         if conv_stats:
#             model.cpu()
#             return model, conv_hist[:epoch+1, :]
#         else:
#             model.cpu()
#             return model
#
#     # --- original single-series path (keep Trainer1 behaviour) ---
#     else:
#         # trainingdata is assumed to be a contiguous time-series (Nsteps, nx)
#         solution = torch.tensor(trainingdata, dtype=torch.float32, device=device)
#         Nsteps, nx = solution.shape
#
#         # Prepare training data (single-step or multistep rollback) - original logic preserved
#         if multistep is None or multistep == 1:
#             inputs = solution[:-1, :].unsqueeze(1)   # (Nsteps-1, 1, nx)
#             targets = solution[1:,  :].unsqueeze(1)   # (Nsteps-1, 1, nx)
#         else:
#             Npredict = Nsteps - multistep
#             inputs = solution[:Npredict, :].unsqueeze(1)
#             targets = torch.stack([solution[t+1 : t+1+multistep, :].unsqueeze(1) for t in range(Npredict)], dim=0)
#
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#         # scheduler handling (preserve compatible defaults)
#         if scheduler == "ReduceLROnPlateau":
#             scheduler_obj = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, verbose=True)
#             scheduler_type = "ReduceLROnPlateau"
#         elif scheduler == "OneCycleLR":
#             # fallback to a simple OneCycle with steps_per_epoch = 1 (single-batch)
#             scheduler_obj = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=1)
#             scheduler_type = "OneCycleLR"
#         elif scheduler == True:
#             scheduler_obj = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
#             scheduler_type = "StepLR"
#         else:
#             scheduler_obj = None
#
#         criterion = nn.MSELoss()
#
#         # plotting / conv_stats as in original Trainer1
#         if plot:
#             fig, ax = plt.subplots(1, 2, figsize=(18, 6))
#             plt.ion(); plt.show()
#
#         if conv_stats:
#             Nsize = 1 + model.N
#             if added_constraints:
#                 Nsize += 3
#             conv_hist = np.zeros([epochs, Nsize])
#
#         # original training loop (full-batch or multistep)
#         for epoch in range(epochs):
#             optimizer.zero_grad(set_to_none=True)
#             model.train()
#
#             if multistep is None or multistep == 1:
#                 predictions = model(inputs)
#             else:
#                 preds = []
#                 u = inputs
#                 for _ in range(multistep):
#                     u = model(u)
#                     preds.append(u)
#                 predictions = torch.stack(preds, dim=1)
#
#             data_loss = criterion(predictions, targets)
#             cl1, cl2, cl3 = (0.0, 0.0, 0.0)
#             if hasattr(model, 'constraint_loss'):
#                 try:
#                     cl1, cl2, cl3 = model.constraint_loss(CFL)
#                 except Exception:
#                     cl1, cl2, cl3 = 0.0, 0.0, 0.0
#
#             if added_constraints:
#                 loss = data_loss + l1 * cl1  + l2 * cl2 + l3 * cl3
#             else:
#                 loss = data_loss
#
#             if plot:
#                 print("Epoch ", epoch , "Loss:", loss.item(),
#                       "weights:", model.conv.weight[:].cpu().detach().numpy())
#
#             loss.backward()
#             optimizer.step()
#             # step scheduler (appropriate to type)
#             if scheduler_obj is not None and scheduler_type == "OneCycleLR":
#                 # OneCycleLR expects per-batch step: we only have one batch -> step here
#                 scheduler_obj.step()
#             elif scheduler_obj is not None and scheduler_type == "StepLR":
#                 scheduler_obj.step()
#             elif scheduler_obj is not None and scheduler_type == "ReduceLROnPlateau":
#                 # pass the data_loss value to the scheduler
#                 scheduler_obj.step(loss.item())
#
#             if conv_stats:
#                 a = np.array(model.conv.weight[:].cpu().detach().numpy()[0][0], dtype=np.float64)
#                 conv_hist[epoch,0] = loss.item()
#                 conv_hist[epoch,1:model.N+1] = a[:]
#                 if added_constraints:
#                     conv_hist[epoch, model.N+1:model.N+1+3] = [float(cl1), float(cl2), float(cl3)]
#
#             if loss.item() < breaktol:
#                 break
#
#             if (epoch + 1) % 100 == 0:
#                 w = model.conv.weight[:].cpu().detach().numpy()[0][0]
#                 j_values = np.arange(-(model.N // 2), (model.N // 2) + 1)
#                 sum_bj = w.sum()
#                 sum_bj_j = (w * j_values).sum()
#                 sum_bj_j2 = (w * j_values**2).sum()
#                 weights = w.copy()/CFL
#                 mid = int(w.shape[0]/2.)
#                 weights[mid] = (w[mid] - 1.)/CFL
#                 print("Epoch ", epoch , "/", epochs,
#                       "Loss:", "{:.3e}".format(loss.item()),
#                       "weights:", np.round(weights,8),
#                       "sum w",  np.round(w.sum(),8),
#                       "sum wj",  np.round((w * j_values).sum(),8),
#                       "2CFL-sum wj^2",  2.*CFL-np.round((w * j_values**2).sum(),8))
#
#         # end original loop
#         if conv_stats:
#             model.cpu()
#             return model, conv_hist[:epoch+1, :]
#         else:
#             model.cpu()
#             return model
#
#
#
# #old trainer
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch._dynamo
#
# torch._dynamo.config.suppress_errors = True
#
# import numpy as np
#
#
# # Training function
# def train_cnn_kernel(model, trainingdata,
#                      dt, dx, mu, epochs=5000, learning_rate=1e-3,
#                      added_constraints=False,
#                      l1=0.0, l2=0.0, l3=0.0,  # Soft constraints on consistency, etc
#                      multistep=False, plot=False,
#                      scheduler=True, breaktol=1e-7,
#                      conv_stats=False, compile_model=False):  # ,
#     # batch_size=64, num_workers=2):
#     """
#     Train the CNNKernelLearner to predict u(t + dt) from u(t).
#     """
#     CFL = mu * dt / dx ** 2
#
#     if added_constraints:
#         l1 = added_constraints[0]
#         l2 = added_constraints[1]
#         l3 = added_constraints[2]
#
#     # Move to device early
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if device:
#         model = model.to(device)
#
#     # Convert to tensor once and move to device
#     solution = torch.tensor(trainingdata, dtype=torch.float32, device=device)
#     Nsteps, nx = solution.shape
#
#     # Compile model for PyTorch 2.0+ (significant speedup)
#     if compile_model and hasattr(torch, 'compile'):
#         try:
#             model = torch.compile(model)
#             print("Model compiled successfully")
#         except Exception as e:
#             print(f"Model compilation failed: {e}")
#
#     # Prepare training data
#     if multistep is None or multistep == 1:
#         # single-step
#         inputs = solution[:-1, :].unsqueeze(1)  # (Nsteps-1, 1, nx)
#         targets = solution[1:, :].unsqueeze(1)  # (Nsteps-1, 1, nx)
#     else:
#         # multi-step rollout
#         Npredict = Nsteps - multistep
#         inputs = solution[:Npredict, :].unsqueeze(1)  # starting states
#         targets = torch.stack(
#             [solution[t + 1: t + 1 + multistep, :].unsqueeze(1)
#              for t in range(Npredict)], dim=0
#         )
#
#     # Loss function and optimizer
#     # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)
#     optimizer = optim.Adam(model.parameters(),
#                            lr=learning_rate)  # , weight_decay=1e-5)  # Add weight decay for regularization
#     if scheduler == "ReduceLROnPlateau":
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                          factor=0.5, patience=200)
#         scheduler == True
#
#     elif scheduler == "OneCycleLR" or scheduler == True:
#         scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
#                                                   epochs=epochs, steps_per_epoch=4)
#         scheduler == True
#     elif scheduler == True:
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
#     else:
#         scheduler == False
#
#     criterion = nn.MSELoss()
#
#     # Setup display and plots
#     if plot:
#         fig, ax = plt.subplots(1, 2, figsize=(18, 6))
#         plt.ion()
#         plt.show()
#
#     if conv_stats:
#         Nsize = 1
#         Nsize += model.N
#         if added_constraints:
#             Nsize += len(added_constraints)
#         conv_hist = np.zeros([epochs, Nsize])
#
#     # Training loop
#     for epoch in range(epochs):
#
#         optimizer.zero_grad(set_to_none=True)
#
#         # Only needed if you have dropout of normalisation
#         # model.train()
#
#         if multistep is None or multistep == 1:
#             predictions = model(inputs)
#
#         else:
#             # rollout k steps
#             preds = []
#             u = inputs
#             for _ in range(multistep):
#                 u = model(u)
#                 preds.append(u)
#
#             # This approach seems faster
#             #            for _ in range(multistep):
#             #                # Set boundary condition
#             #                u[:, :, 0] = bBC
#             #                u[:, :, -1] = tBC
#             #                kernel = model.conv.weight[:]
#             #                u = F.conv1d(u, kernel, padding=1)
#             #                preds.append(u)  # Save field history
#
#             predictions = torch.stack(preds, dim=1)  # (Npredict, multistep, 1, nx)
#
#         data_loss = criterion(predictions, targets)
#
#         # Constraint losses
#         cl1, cl2, cl3 = model.constraint_loss(CFL)
#         if added_constraints:
#             loss = data_loss + l1 * cl1 + l2 * cl2 + l3 * cl3
#         else:
#             loss = data_loss
#
#         if plot:
#             print("Epoch ", epoch, "Loss:", loss.item(),
#                   "weights:", model.conv.weight[:].cpu().detach().numpy())
#
#             if multistep:
#                 ax[0].pcolormesh(predictions[-2 * Nsteps:, 0, :].cpu().detach().numpy())
#                 ax[1].pcolormesh(targets[-2 * Nsteps:, 0, :].cpu().detach().numpy())
#             else:
#                 ax[0].pcolormesh(predictions[:, 0, :].cpu().detach().numpy())
#                 ax[1].pcolormesh(targets[:, 0, :].cpu().detach().numpy())
#             plt.pause(0.001)
#             ax[0].cla()
#             ax[1].cla()
#
#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()
#         if scheduler:
#             scheduler.step()
#
#         # print(np.sum(model.conv.weight[:].detach().numpy()[0][0]))
#
#         if conv_stats:
#             a = np.array(model.conv.weight[:].cpu().detach().numpy()[0][0], dtype=np.float64)
#             conv_hist[epoch, 0] = loss.item()
#             conv_hist[epoch, 1:model.N + 1] = a[:]
#             if added_constraints:
#                 conv_hist[epoch, model.N + 1:model.N + 1 + len(added_constraints)] = [cl1.cpu().detach().numpy(),
#                                                                                       cl2.cpu().detach().numpy(),
#                                                                                       cl3.cpu().detach().numpy()]
#
#             # print(conv_hist[epoch,:])
#
#         # Add in break condition for low error
#         if loss.item() < breaktol:
#             break
#
#         # Print loss every 100 epochs
#         if (epoch + 1) % 100 == 0:
#
#             w = model.conv.weight[:].cpu().detach().numpy()[0][0]
#             j_values = np.arange(-(model.N // 2),
#                                  (model.N // 2) + 1)
#             # constraints
#             sum_bj = w.sum()
#             sum_bj_j = (w * j_values).sum()
#             sum_bj_j2 = (w * j_values ** 2).sum()
#
#             weights = np.zeros(w.shape[0])
#             weights = w.copy() / CFL
#             mid = int(w.shape[0] / 2.)
#             weights[mid] = (w[mid] - 1.) / CFL
#
#             print("Epoch ", epoch, "/", epochs,
#                   "Loss:", "{:.3e}".format(loss.item()),
#                   "weights:", np.round(weights, 8),
#                   "sum w", np.round(w.sum(), 8),
#                   "sum wj", np.round((w * j_values).sum(), 8),
#                   "2CFL-sum wj^2", 2. * CFL - np.round((w * j_values ** 2).sum(), 8))
#             if added_constraints:
#                 print("constraints", cl1.cpu().detach().numpy(),
#                       cl2.cpu().detach().numpy(),
#                       cl3.cpu().detach().numpy())
#
#     if device:
#         model.cpu()
#         model.device = "cpu"
#
#     if conv_stats:
#         return model, conv_hist[:epoch + 1, :]
#     else:
#         return model



# #modified trainer 1 gpt
# import inspect
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
#
# import torch.utils.data as data_utils
#
#
#
# def train_cnn_kernel(model, trainingdata,
#                      dt, dx, mu, epochs=5000, learning_rate=1e-3,
#                      added_constraints=False,
#                      l1=0.0, l2=0.0, l3=0.0,  # Soft constraints on consistency, etc
#                      multistep=False, plot=False,
#                      scheduler=True, breaktol=1e-7,
#                      conv_stats=False, compile_model=False,
#                      multicase=False, batch_size=None, shuffle=False, **kwargs):
#     """
#     Train the CNNKernelLearner to predict u(t + dt) from u(t).
#
#     Minimal changes to support:
#       - `trainingdata` as a tuple (inputs, targets, bcs) for multicase use,
#       - optional batching and shuffling via batch_size and shuffle.
#     """
#     CFL = mu * dt / dx ** 2
#
#     if added_constraints:
#         l1 = added_constraints[0]
#         l2 = added_constraints[1]
#         l3 = added_constraints[2]
#
#     # Move to device early
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if device:
#         model = model.to(device)
#
#     # If trainingdata is provided as tuple/list (multicase): unpack as sample pairs
#     use_pairs = False
#     dataloader = None
#     if isinstance(trainingdata, (tuple, list)):
#         # Expect (inputs, targets, ...) ; ignore extra items (e.g., bcs) if present
#         inputs_np = trainingdata[0]
#         targets_np = trainingdata[1]
#         # Convert to tensors and add channel dim
#         inputs = torch.tensor(inputs_np, dtype=torch.float32).unsqueeze(1).to(device)   # (N,1,nx)
#         targets = torch.tensor(targets_np, dtype=torch.float32).unsqueeze(1).to(device) # (N,1,nx)
#         use_pairs = True
#
#         # Make a DataLoader if batch_size is specified, else use full-batch
#         if batch_size is None:
#             batch_size = inputs.shape[0]
#         dataset = data_utils.TensorDataset(inputs, targets)
#         dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
#                                            shuffle=bool(shuffle))
#     else:
#         # Convert to tensor once and move to device (old behaviour)
#         solution = torch.tensor(trainingdata, dtype=torch.float32, device=device)
#         Nsteps, nx = solution.shape
#
#     # Compile model for PyTorch 2.0+ (significant speedup)
#     if compile_model and hasattr(torch, 'compile'):
#         try:
#             model = torch.compile(model)
#             print("Model compiled successfully")
#         except Exception as e:
#             print(f"Model compilation failed: {e}")
#
#     # Prepare training data for the time-series path (unchanged)
#     if not use_pairs:
#         if multistep is None or multistep == 1:
#             # single-step
#             inputs_ts = solution[:-1, :].unsqueeze(1)  # (Nsteps-1, 1, nx)
#             targets_ts = solution[1:, :].unsqueeze(1)  # (Nsteps-1, 1, nx)
#         else:
#             # multi-step rollout
#             Npredict = Nsteps - multistep
#             inputs_ts = solution[:Npredict, :].unsqueeze(1)  # starting states
#             targets_ts = torch.stack(
#                 [solution[t + 1: t + 1 + multistep, :].unsqueeze(1)
#                  for t in range(Npredict)], dim=0
#             )
#
#     # Loss function and optimizer
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     if scheduler == "ReduceLROnPlateau":
#         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                          factor=0.5, patience=200)
#         scheduler == True
#
#     elif scheduler == "OneCycleLR" or scheduler == True:
#         scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
#                                                   epochs=epochs, steps_per_epoch=max(1, (inputs.shape[0] if use_pairs else 4) // (batch_size or 1)))
#         scheduler == True
#     elif scheduler == True:
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
#     else:
#         scheduler == False
#
#     criterion = nn.MSELoss()
#
#     # Setup display and plots
#     if plot:
#         import matplotlib.pyplot as plt
#         fig, ax = plt.subplots(1, 2, figsize=(18, 6))
#         plt.ion()
#         plt.show()
#
#     if conv_stats:
#         Nsize = 1
#         Nsize += model.N
#         if added_constraints:
#             Nsize += len(added_constraints)
#         conv_hist = np.zeros([epochs, Nsize])
#
#     # Training loop
#     for epoch in range(epochs):
#
#         # If using pairs (DataLoader), iterate batches
#         if use_pairs:
#             epoch_losses = []
#             for batch_inputs, batch_targets in dataloader:
#                 optimizer.zero_grad(set_to_none=True)
#
#                 # forward
#                 if multistep is None or multistep == 1:
#                     predictions = model(batch_inputs)
#                 else:
#                     # rollout k steps
#                     preds = []
#                     u = batch_inputs
#                     for _ in range(multistep):
#                         u = model(u)
#                         preds.append(u)
#                     predictions = torch.stack(preds, dim=1)  # (Nbatch, multistep, 1, nx)
#
#                 data_loss = criterion(predictions, batch_targets)
#
#                 # Constraint losses
#                 cl1, cl2, cl3 = model.constraint_loss(CFL)
#                 if added_constraints:
#                     loss = data_loss + l1 * cl1 + l2 * cl2 + l3 * cl3
#                 else:
#                     loss = data_loss
#
#                 loss.backward()
#                 optimizer.step()
#                 epoch_losses.append(loss.item())
#
#             # Scheduler step for epoch-based schedulers
#             if scheduler:
#                 # ReduceLROnPlateau expects metric
#                 if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
#                     scheduler.step(np.mean(epoch_losses))
#                 else:
#                     try:
#                         scheduler.step()
#                     except Exception:
#                         pass
#
#             # record conv_stats using model's current kernel and mean loss
#             if conv_stats:
#                 mean_loss = float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else 0.0
#                 a = np.array(model.conv.weight[:].cpu().detach().numpy()[0][0], dtype=np.float64)
#                 conv_hist[epoch, 0] = mean_loss
#                 conv_hist[epoch, 1:model.N + 1] = a[:]
#                 if added_constraints:
#                     conv_hist[epoch, model.N + 1:model.N + 1 + len(added_constraints)] = [cl1.cpu().detach().numpy(),
#                                                                                           cl2.cpu().detach().numpy(),
#                                                                                           cl3.cpu().detach().numpy()]
#
#             # break condition on averaged epoch loss
#             if np.mean(epoch_losses) < breaktol:
#                 break
#
#         else:
#             # original timeseries path (unchanged except variable names)
#             optimizer.zero_grad(set_to_none=True)
#
#             if multistep is None or multistep == 1:
#                 predictions = model(inputs_ts)
#             else:
#                 preds = []
#                 u = inputs_ts
#                 for _ in range(multistep):
#                     u = model(u)
#                     preds.append(u)
#                 predictions = torch.stack(preds, dim=1)  # (Npredict, multistep, 1, nx)
#
#             data_loss = criterion(predictions, targets_ts)
#
#             # Constraint losses
#             cl1, cl2, cl3 = model.constraint_loss(CFL)
#             if added_constraints:
#                 loss = data_loss + l1 * cl1 + l2 * cl2 + l3 * cl3
#             else:
#                 loss = data_loss
#
#             if plot:
#                 import matplotlib.pyplot as plt
#                 print("Epoch ", epoch, "Loss:", loss.item(),
#                       "weights:", model.conv.weight[:].cpu().detach().numpy())
#
#                 if multistep:
#                     ax[0].pcolormesh(predictions[-2 * Nsteps:, 0, :].cpu().detach().numpy())
#                     ax[1].pcolormesh(targets_ts[-2 * Nsteps:, 0, :].cpu().detach().numpy())
#                 else:
#                     ax[0].pcolormesh(predictions[:, 0, :].cpu().detach().numpy())
#                     ax[1].pcolormesh(targets_ts[:, 0, :].cpu().detach().numpy())
#                 plt.pause(0.001)
#                 ax[0].cla()
#                 ax[1].cla()
#
#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()
#             if scheduler:
#                 # for ReduceLROnPlateau we need a metric, fallback to loss.item()
#                 if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
#                     scheduler.step(loss.item())
#                 else:
#                     try:
#                         scheduler.step()
#                     except Exception:
#                         pass
#
#             if conv_stats:
#                 a = np.array(model.conv.weight[:].cpu().detach().numpy()[0][0], dtype=np.float64)
#                 conv_hist[epoch, 0] = loss.item()
#                 conv_hist[epoch, 1:model.N + 1] = a[:]
#                 if added_constraints:
#                     conv_hist[epoch, model.N + 1:model.N + 1 + len(added_constraints)] = [cl1.cpu().detach().numpy(),
#                                                                                           cl2.cpu().detach().numpy(),
#                                                                                           cl3.cpu().detach().numpy()]
#
#             # Add in break condition for low error
#             if loss.item() < breaktol:
#                 break
#
#         # Print loss every 100 epochs (keeps previous print behaviour)
#         if (epoch + 1) % 100 == 0:
#             w = model.conv.weight[:].cpu().detach().numpy()[0][0]
#             j_values = np.arange(-(model.N // 2),
#                                  (model.N // 2) + 1)
#             # constraints
#             sum_bj = w.sum()
#             sum_bj_j = (w * j_values).sum()
#             sum_bj_j2 = (w * j_values ** 2).sum()
#
#             weights = np.zeros(w.shape[0])
#             weights = w.copy() / CFL
#             mid = int(w.shape[0] / 2.)
#             weights[mid] = (w[mid] - 1.) / CFL
#
#             print("Epoch ", epoch, "/", epochs,
#                   "Loss:", "{:.3e}".format(conv_hist[epoch, 0] if conv_stats else (loss.item() if not use_pairs else np.mean(epoch_losses))),
#                   "weights:", np.round(weights, 8),
#                   "sum w", np.round(w.sum(), 8),
#                   "sum wj", np.round((w * j_values).sum(), 8),
#                   "2CFL-sum wj^2", 2. * CFL - np.round((w * j_values ** 2).sum(), 8))
#             if added_constraints:
#                 print("constraints", cl1.cpu().detach().numpy(),
#                       cl2.cpu().detach().numpy(),
#                       cl3.cpu().detach().numpy())
#
#     if device:
#         model.cpu()
#         model.device = "cpu"
#
#     if conv_stats:
#         return model, conv_hist[:epoch + 1, :]
#     else:
#         return model



#modified trainer 2 claude
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch._dynamo
#
# torch._dynamo.config.suppress_errors = True
#
# import numpy as np
#
#
# # Training function
# def train_cnn_kernel(model, trainingdata,
#                      dt, dx, mu, epochs=5000, learning_rate=1e-3,
#                      added_constraints=False,
#                      l1=0.0, l2=0.0, l3=0.0,  # Soft constraints on consistency, etc
#                      multistep=False, plot=False,
#                      scheduler=True, breaktol=1e-7,
#                      conv_stats=False, compile_model=False,
#                      multicase=False, batch_size=None, shuffle=False):
#     """
#     Train the CNNKernelLearner to predict u(t + dt) from u(t).
#
#     Args:
#         multicase: If True, trainingdata should be tuple of (inputs, targets, bcs)
#         batch_size: Batch size for multicase training (default: use all data)
#         shuffle: Whether to shuffle data in multicase training
#     """
#     CFL = mu * dt / dx ** 2
#
#     if added_constraints:
#         l1 = added_constraints[0]
#         l2 = added_constraints[1]
#         l3 = added_constraints[2]
#
#     # Move to device early
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if device:
#         model = model.to(device)
#
#     # Handle multicase vs single case data
#     if multicase:
#         # trainingdata is a tuple: (inputs, targets, bcs)
#         inputs_np, targets_np, bcs_np = trainingdata
#         inputs = torch.tensor(inputs_np, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1, nx)
#         targets = torch.tensor(targets_np, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1, nx)
#         bcs = torch.tensor(bcs_np, dtype=torch.float32, device=device)  # (N, 2)
#
#         # Create dataset and dataloader if batch_size specified
#         if batch_size is not None:
#             dataset = torch.utils.data.TensorDataset(inputs, targets, bcs)
#             dataloader = torch.utils.data.DataLoader(
#                 dataset, batch_size=batch_size, shuffle=shuffle
#             )
#         else:
#             dataloader = None
#     else:
#         # Original behavior: trainingdata is 2D array (Nsteps, nx)
#         solution = torch.tensor(trainingdata, dtype=torch.float32, device=device)
#         Nsteps, nx = solution.shape
#
#         # Prepare training data
#         if multistep is None or multistep == 1:
#             # single-step
#             inputs = solution[:-1, :].unsqueeze(1)  # (Nsteps-1, 1, nx)
#             targets = solution[1:, :].unsqueeze(1)  # (Nsteps-1, 1, nx)
#         else:
#             # multi-step rollout
#             Npredict = Nsteps - multistep
#             inputs = solution[:Npredict, :].unsqueeze(1)  # starting states
#             targets = torch.stack(
#                 [solution[t + 1: t + 1 + multistep, :].unsqueeze(1)
#                  for t in range(Npredict)], dim=0
#             )
#         dataloader = None
#
#     # Compile model for PyTorch 2.0+ (significant speedup)
#     if compile_model and hasattr(torch, 'compile'):
#         try:
#             model = torch.compile(model)
#             print("Model compiled successfully")
#         except Exception as e:
#             print(f"Model compilation failed: {e}")
#
#     # Loss function and optimizer
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#     if scheduler == "ReduceLROnPlateau":
#         scheduler_obj = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                              factor=0.5, patience=200)
#     elif scheduler == "OneCycleLR" or scheduler == True:
#         steps_per_epoch = len(dataloader) if dataloader else 4
#         scheduler_obj = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
#                                                       epochs=epochs, steps_per_epoch=steps_per_epoch)
#     elif scheduler:
#         scheduler_obj = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
#     else:
#         scheduler_obj = None
#
#     criterion = nn.MSELoss()
#
#     # Setup display and plots
#     if plot:
#         import matplotlib.pyplot as plt
#         fig, ax = plt.subplots(1, 2, figsize=(18, 6))
#         plt.ion()
#         plt.show()
#
#     if conv_stats:
#         Nsize = 1
#         Nsize += model.N
#         if added_constraints:
#             Nsize += len(added_constraints)
#         conv_hist = np.zeros([epochs, Nsize])
#
#     # Training loop
#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         n_batches = 0
#
#         if dataloader:
#             # Batch training for multicase
#             for batch_inputs, batch_targets, batch_bcs in dataloader:
#                 optimizer.zero_grad(set_to_none=True)
#
#                 # Forward pass
#                 predictions = model(batch_inputs)
#
#                 data_loss = criterion(predictions, batch_targets)
#
#                 # Constraint losses
#                 cl1, cl2, cl3 = model.constraint_loss(CFL)
#                 if added_constraints:
#                     loss = data_loss + l1 * cl1 + l2 * cl2 + l3 * cl3
#                 else:
#                     loss = data_loss
#
#                 # Backward pass
#                 loss.backward()
#                 optimizer.step()
#
#                 epoch_loss += loss.item()
#                 n_batches += 1
#
#             # Average loss over batches
#             loss = epoch_loss / n_batches
#
#             # Scheduler step (once per epoch)
#             if scheduler_obj and scheduler_obj.__class__.__name__ != 'OneCycleLR':
#                 if isinstance(scheduler_obj, optim.lr_scheduler.ReduceLROnPlateau):
#                     scheduler_obj.step(loss)
#                 else:
#                     scheduler_obj.step()
#         else:
#             # Original single-batch training
#             optimizer.zero_grad(set_to_none=True)
#
#             if multistep is None or multistep == 1:
#                 predictions = model(inputs)
#             else:
#                 # rollout k steps
#                 preds = []
#                 u = inputs
#                 for _ in range(multistep):
#                     u = model(u)
#                     preds.append(u)
#                 predictions = torch.stack(preds, dim=1)  # (Npredict, multistep, 1, nx)
#
#             data_loss = criterion(predictions, targets)
#
#             # Constraint losses
#             cl1, cl2, cl3 = model.constraint_loss(CFL)
#             if added_constraints:
#                 loss = data_loss + l1 * cl1 + l2 * cl2 + l3 * cl3
#             else:
#                 loss = data_loss
#
#             if plot:
#                 print("Epoch ", epoch, "Loss:", loss.item(),
#                       "weights:", model.conv.weight[:].cpu().detach().numpy())
#
#                 if multistep:
#                     ax[0].pcolormesh(predictions[-2 * Nsteps:, 0, :].cpu().detach().numpy())
#                     ax[1].pcolormesh(targets[-2 * Nsteps:, 0, :].cpu().detach().numpy())
#                 else:
#                     ax[0].pcolormesh(predictions[:, 0, :].cpu().detach().numpy())
#                     ax[1].pcolormesh(targets[:, 0, :].cpu().detach().numpy())
#                 plt.pause(0.001)
#                 ax[0].cla()
#                 ax[1].cla()
#
#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()
#             if scheduler_obj:
#                 if isinstance(scheduler_obj, optim.lr_scheduler.ReduceLROnPlateau):
#                     scheduler_obj.step(loss.item())
#                 else:
#                     scheduler_obj.step()
#
#         if conv_stats:
#             a = np.array(model.conv.weight[:].cpu().detach().numpy()[0][0], dtype=np.float64)
#             conv_hist[epoch, 0] = loss if isinstance(loss, float) else loss.item()
#             conv_hist[epoch, 1:model.N + 1] = a[:]
#             if added_constraints:
#                 conv_hist[epoch, model.N + 1:model.N + 1 + len(added_constraints)] = [
#                     cl1.cpu().detach().numpy(),
#                     cl2.cpu().detach().numpy(),
#                     cl3.cpu().detach().numpy()
#                 ]
#
#         # Add in break condition for low error
#         loss_value = loss if isinstance(loss, float) else loss.item()
#         if loss_value < breaktol:
#             break
#
#         # Print loss every 100 epochs
#         if (epoch + 1) % 100 == 0:
#             w = model.conv.weight[:].cpu().detach().numpy()[0][0]
#             j_values = np.arange(-(model.N // 2), (model.N // 2) + 1)
#
#             weights = np.zeros(w.shape[0])
#             weights = w.copy() / CFL
#             mid = int(w.shape[0] / 2.)
#             weights[mid] = (w[mid] - 1.) / CFL
#
#             print("Epoch ", epoch, "/", epochs,
#                   "Loss:", "{:.3e}".format(loss_value),
#                   "weights:", np.round(weights, 8),
#                   "sum w", np.round(w.sum(), 8),
#                   "sum wj", np.round((w * j_values).sum(), 8),
#                   "2CFL-sum wj^2", 2. * CFL - np.round((w * j_values ** 2).sum(), 8))
#             if added_constraints:
#                 print("constraints", cl1.cpu().detach().numpy(),
#                       cl2.cpu().detach().numpy(),
#                       cl3.cpu().detach().numpy())
#
#     if device:
#         model.cpu()
#         model.device = "cpu"
#
#     if conv_stats:
#         return model, conv_hist[:epoch + 1, :]
#     else:
#         return model



# #11th March 2026
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch._dynamo
#
# torch._dynamo.config.suppress_errors = True
#
# import numpy as np
#
#
# # Training function
# def train_cnn_kernel(model, trainingdata,
#                      dt, dx, mu, epochs=5000, learning_rate=1e-3,
#                      added_constraints=False,
#                      l1=0.0, l2=0.0, l3=0.0,  # Soft constraints on consistency, etc
#                      multistep=False, plot=False,
#                      scheduler=True, breaktol=1e-7,
#                      conv_stats=False, compile_model=False,
#                      multicase=False, batch_size=None, shuffle=False,
#                      forcing_history=None, grad_clip=0.0):
#     """
#     Train the CNNKernelLearner to predict u(t + dt) from u(t).
#
#     Args:
#         multicase: If True, trainingdata should be tuple of (inputs, targets, bcs)
#         batch_size: Batch size for multicase training (default: use all data)
#         shuffle: Whether to shuffle data in multicase training
#         forcing_history: Optional forcing array (T, nx) or callable(n) for Burgers equation.
#                          When provided and model.in_ch >= 3, assembles [u, u²/2, f] input
#                          once before training — no per-epoch overhead.
#         grad_clip: Max-norm for gradient clipping. 0 = disabled.
#     """
#     CFL = mu * dt / dx ** 2
#
#     if added_constraints:
#         l1 = added_constraints[0]
#         l2 = added_constraints[1]
#         l3 = added_constraints[2]
#
#     # Move to device early
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if device:
#         model = model.to(device)
#
#     # Handle multicase vs single case data
#     if multicase:
#         # trainingdata is a tuple: (inputs, targets, bcs)
#         inputs_np, targets_np, bcs_np = trainingdata
#         inputs = torch.tensor(inputs_np, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1, nx)
#         targets = torch.tensor(targets_np, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1, nx)
#         bcs = torch.tensor(bcs_np, dtype=torch.float32, device=device)  # (N, 2)
#
#         # Create dataset and dataloader if batch_size specified
#         if batch_size is not None:
#             dataset = torch.utils.data.TensorDataset(inputs, targets, bcs)
#             dataloader = torch.utils.data.DataLoader(
#                 dataset, batch_size=batch_size, shuffle=shuffle
#             )
#         else:
#             dataloader = None
#     else:
#         # Original behavior: trainingdata is 2D array (Nsteps, nx)
#         solution = torch.tensor(trainingdata, dtype=torch.float32, device=device)
#         Nsteps, nx = solution.shape
#
#         # Prepare training data
#         if not multistep or multistep == 1:
#             # single-step
#             inputs = solution[:-1, :].unsqueeze(1)   # (Nsteps-1, 1, nx)
#             targets = solution[1:, :].unsqueeze(1)   # (Nsteps-1, 1, nx)
#         else:
#             # multi-step rollout
#             Npredict = Nsteps - multistep
#             inputs = solution[:Npredict, :].unsqueeze(1)
#             targets = torch.stack(
#                 [solution[t + 1: t + 1 + multistep, :].unsqueeze(1)
#                  for t in range(Npredict)], dim=0
#             )
#         dataloader = None
#
#     # ------------------------------------------------------------------ #
#     # Pre-assemble multi-channel inputs ONCE before the training loop.    #
#     # This matches the behaviour of the old train_burgers_kernel which     #
#     # called prepare_training_data_burgers() upfront — no per-epoch cost. #
#     # ------------------------------------------------------------------ #
#     use_multichannel = (
#         not multicase
#         and (not multistep or multistep == 1)
#         and hasattr(model, 'in_ch')
#         and model.in_ch >= 2
#     )
#
#     if use_multichannel:
#         u_sq = 0.5 * inputs * inputs   # (T-1, 1, nx)
#
#         if model.in_ch >= 3 and forcing_history is not None:
#             # Build forcing tensor once
#             if callable(forcing_history):
#                 f_list = [
#                     torch.tensor(forcing_history(n), dtype=torch.float32, device=device)
#                     .unsqueeze(0).unsqueeze(0)
#                     for n in range(inputs.shape[0])
#                 ]
#                 f_stack = torch.cat(f_list, dim=0)          # (T-1, 1, nx)
#             else:
#                 forcing_arr = np.asarray(forcing_history, dtype=np.float32)
#                 f_np = forcing_arr[:inputs.shape[0]]
#                 if f_np.ndim == 1:
#                     f_np = np.tile(f_np, (inputs.shape[0], 1))
#                 f_stack = torch.tensor(f_np, dtype=torch.float32, device=device).unsqueeze(1)
#
#             inputs = torch.cat([inputs, u_sq, f_stack], dim=1)   # (T-1, 3, nx)
#
#         elif model.in_ch == 2:
#             inputs = torch.cat([inputs, u_sq], dim=1)             # (T-1, 2, nx)
#
#     # Compile model for PyTorch 2.0+ (significant speedup)
#     if compile_model and hasattr(torch, 'compile'):
#         try:
#             model = torch.compile(model)
#             print("Model compiled successfully")
#         except Exception as e:
#             print(f"Model compilation failed: {e}")
#
#     # Loss function and optimizer
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#     if scheduler == "ReduceLROnPlateau":
#         scheduler_obj = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                              factor=0.5, patience=50)
#     elif scheduler == "OneCycleLR" or scheduler == True:
#         steps_per_epoch = len(dataloader) if dataloader else 4
#         scheduler_obj = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
#                                                       epochs=epochs, steps_per_epoch=steps_per_epoch)
#     elif scheduler:
#         scheduler_obj = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
#     else:
#         scheduler_obj = None
#
#     criterion = nn.MSELoss()
#
#     # Setup display and plots
#     if plot:
#         import matplotlib.pyplot as plt
#         fig, ax = plt.subplots(1, 2, figsize=(18, 6))
#         plt.ion()
#         plt.show()
#
#     if conv_stats:
#         Nsize = 1 + model.N
#         if added_constraints:
#             Nsize += len(added_constraints)
#         conv_hist = np.zeros([epochs, Nsize])
#
#     # Training loop
#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         n_batches = 0
#
#         if dataloader:
#             # Batch training for multicase
#             for batch_inputs, batch_targets, batch_bcs in dataloader:
#                 optimizer.zero_grad(set_to_none=True)
#
#                 predictions = model(batch_inputs)
#                 data_loss = criterion(predictions, batch_targets)
#
#                 cl1, cl2, cl3 = model.constraint_loss(CFL)
#                 if added_constraints:
#                     loss = data_loss + l1 * cl1 + l2 * cl2 + l3 * cl3
#                 else:
#                     loss = data_loss
#
#                 # NaN/Inf check
#                 if torch.isnan(loss) or torch.isinf(loss):
#                     print(f"NaN/Inf detected at epoch {epoch} — stopping.")
#                     break
#
#                 loss.backward()
#                 if grad_clip > 0:
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
#                 optimizer.step()
#
#                 epoch_loss += loss.item()
#                 n_batches += 1
#
#             loss = epoch_loss / n_batches
#
#             if scheduler_obj and scheduler_obj.__class__.__name__ != 'OneCycleLR':
#                 if isinstance(scheduler_obj, optim.lr_scheduler.ReduceLROnPlateau):
#                     scheduler_obj.step(loss)
#                 else:
#                     scheduler_obj.step()
#         else:
#             # Single-batch training
#             optimizer.zero_grad(set_to_none=True)
#
#             if not multistep or multistep == 1:
#                 # inputs already has the right channels (pre-assembled above)
#                 predictions = model(inputs)
#             else:
#                 # rollout k steps
#                 preds = []
#                 u = inputs
#                 for step in range(multistep):
#                     u = model(u)
#                     preds.append(u)
#                 predictions = torch.stack(preds, dim=1)  # (Npredict, multistep, 1, nx)
#
#             data_loss = criterion(predictions, targets)
#
#             cl1, cl2, cl3 = model.constraint_loss(CFL)
#             if added_constraints:
#                 loss = data_loss + l1 * cl1 + l2 * cl2 + l3 * cl3
#             else:
#                 loss = data_loss
#
#             if plot:
#                 print("Epoch ", epoch, "Loss:", loss.item(),
#                       "weights:", model.conv.weight[:].cpu().detach().numpy())
#                 if multistep:
#                     ax[0].pcolormesh(predictions[-2 * Nsteps:, 0, :].cpu().detach().numpy())
#                     ax[1].pcolormesh(targets[-2 * Nsteps:, 0, :].cpu().detach().numpy())
#                 else:
#                     ax[0].pcolormesh(predictions[:, 0, :].cpu().detach().numpy())
#                     ax[1].pcolormesh(targets[:, 0, :].cpu().detach().numpy())
#                 plt.pause(0.001)
#                 ax[0].cla()
#                 ax[1].cla()
#
#             # NaN/Inf check
#             if torch.isnan(loss) or torch.isinf(loss):
#                 print(f"NaN/Inf detected at epoch {epoch} — stopping.")
#                 break
#
#             loss.backward()
#             if grad_clip > 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
#             optimizer.step()
#
#             if scheduler_obj:
#                 if isinstance(scheduler_obj, optim.lr_scheduler.ReduceLROnPlateau):
#                     scheduler_obj.step(loss.item())
#                 else:
#                     scheduler_obj.step()
#
#         if conv_stats:
#             a = np.array(model.conv.weight[:].cpu().detach().numpy()[0][0], dtype=np.float64)
#             conv_hist[epoch, 0] = loss if isinstance(loss, float) else loss.item()
#             conv_hist[epoch, 1:model.N + 1] = a[:]
#             if added_constraints:
#                 conv_hist[epoch, model.N + 1:model.N + 1 + len(added_constraints)] = [
#                     cl1.cpu().detach().numpy(),
#                     cl2.cpu().detach().numpy(),
#                     cl3.cpu().detach().numpy()
#                 ]
#
#         # Break condition for low error
#         loss_value = loss if isinstance(loss, float) else loss.item()
#         if loss_value < breaktol:
#             break
#
#         # Print loss every 100 epochs
#         if (epoch + 1) % 100 == 0:
#             w = model.conv.weight[:].cpu().detach().numpy()[0][0]
#             j_values = np.arange(-(model.N // 2), (model.N // 2) + 1)
#
#             weights = w.copy() / CFL
#             mid = int(w.shape[0] / 2.)
#             weights[mid] = (w[mid] - 1.) / CFL
#
#             print("Epoch ", epoch, "/", epochs,
#                   "Loss:", "{:.3e}".format(loss_value),
#                   "weights:", np.round(weights, 8),
#                   "sum w", np.round(w.sum(), 8),
#                   "sum wj", np.round((w * j_values).sum(), 8),
#                   "2CFL-sum wj^2", 2. * CFL - np.round((w * j_values ** 2).sum(), 8))
#             if added_constraints:
#                 print("constraints", cl1.cpu().detach().numpy(),
#                       cl2.cpu().detach().numpy(),
#                       cl3.cpu().detach().numpy())
#
#     if device:
#         model.cpu()
#         model.device = "cpu"
#
#     if conv_stats:
#         return model, conv_hist[:epoch + 1, :]
#     else:
#         return model




import torch
import torch.nn as nn
import torch.optim as optim
import torch._dynamo

torch._dynamo.config.suppress_errors = True

import numpy as np


# Training function
def train_cnn_kernel(model, trainingdata,
                     dt, dx, mu, epochs=5000, learning_rate=1e-3,
                     added_constraints=False,
                     l1=0.0, l2=0.0, l3=0.0,
                     multistep=False, plot=False,
                     scheduler=True, breaktol=1e-7,
                     conv_stats=False, compile_model=False,
                     multicase=False, batch_size=None, shuffle=False,
                     forcing_history=None, grad_clip=0.0,
                     velocity=None,
                     left_bc=None,
                     right_bc=None,
                     transport=False,
                     loss_on_interior=False,
                     print_every_transport=1000):
    """
    Train the CNNKernelLearner to predict u(t + dt) from u(t).

    Args:
        multicase: If True, trainingdata should be tuple of (inputs, targets, bcs)
        batch_size: Batch size for multicase training (default: use all data)
        shuffle: Whether to shuffle data in multicase training
        forcing_history: Optional forcing array (T, nx) or callable(n) for Burgers equation.
                         When provided and model.in_ch >= 3, assembles [u, u²/2, f] input
                         once before training — no per-epoch overhead.
        grad_clip: Max-norm for gradient clipping. 0 = disabled.
    """
    CFL = mu * dt / dx ** 2

    if added_constraints:
        l1 = added_constraints[0]
        l2 = added_constraints[1]
        l3 = added_constraints[2]

    # Move to device early
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device:
        model = model.to(device)

    # ------------------------------------------------------------------ #
    # TRANSPORT BRANCH
    # ------------------------------------------------------------------ #
    # This branch trains TransportScalarKernelLearner:
    #
    #     Gamma_t + d(Gamma*u_s)/dx = D_s Gamma_xx
    #
    # Required:
    #     trainingdata = concentration, shape (Nt, Nx) or (Nx, Nt)
    #     velocity     = tangential velocity, same shape
    #     left_bc      = C_left_boundary, shape (Nt,)
    #     right_bc     = C_right_boundary, shape (Nt,)
    #
    # It returns:
    #     model, hist
    #
    # where hist columns are:
    #     [loss, alpha, beta, Ds]
    # ------------------------------------------------------------------ #

    is_transport_model = (
        transport
        or velocity is not None
        or (
            hasattr(model, "get_Ds")
            and hasattr(model, "get_kernels")
            and not hasattr(model, "conv")
        )
    )

    if is_transport_model:

        if velocity is None:
            raise ValueError("For transport training, provide velocity=velocity_train.")

        if left_bc is None or right_bc is None:
            raise ValueError("For transport training, provide left_bc and right_bc.")

        C = np.asarray(trainingdata, dtype=np.float32)
        U = np.asarray(velocity, dtype=np.float32)
        LBC = np.asarray(left_bc, dtype=np.float32)
        RBC = np.asarray(right_bc, dtype=np.float32)

        # Prefer shape (Nt, Nx). If user passed (Nx, Nt), transpose.
        if C.ndim != 2:
            raise ValueError(f"transport concentration must be 2D, got shape {C.shape}")

        if U.ndim != 2:
            raise ValueError(f"transport velocity must be 2D, got shape {U.shape}")

        if C.shape[0] != LBC.shape[0] and C.shape[1] == LBC.shape[0]:
            C = C.T

        if U.shape[0] != LBC.shape[0] and U.shape[1] == LBC.shape[0]:
            U = U.T

        if C.shape != U.shape:
            raise ValueError(
                f"concentration and velocity must have same shape; "
                f"got {C.shape} and {U.shape}"
            )

        if C.shape[0] != LBC.shape[0] or C.shape[0] != RBC.shape[0]:
            raise ValueError(
                f"time dimension mismatch: C has Nt={C.shape[0]}, "
                f"left_bc={LBC.shape[0]}, right_bc={RBC.shape[0]}"
            )

        C_tensor = torch.tensor(C, dtype=torch.float32, device=device)
        U_tensor = torch.tensor(U, dtype=torch.float32, device=device)
        LBC_tensor = torch.tensor(LBC, dtype=torch.float32, device=device)
        RBC_tensor = torch.tensor(RBC, dtype=torch.float32, device=device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        if scheduler == "ReduceLROnPlateau":
            scheduler_obj = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=100
            )
        elif scheduler:
            scheduler_obj = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=1000,
                gamma=0.95
            )
        else:
            scheduler_obj = None

        hist = []

        print("\n" + "=" * 80)
        print("Training transport kernel")
        print("=" * 80)

        for epoch in range(epochs):

            optimizer.zero_grad(set_to_none=True)

            gamma_n = C_tensor[:-1, :]
            u_n = U_tensor[:-1, :]
            gamma_np1 = C_tensor[1:, :]

            # Shape: (Nt-1, 2, Nx)
            x_in = torch.stack([gamma_n, u_n], dim=1)

            pred = model(
                x_in,
                left_bc_pad=LBC_tensor[:-1],
                right_bc_pad=RBC_tensor[:-1],
                left_bc_out=LBC_tensor[1:],
                right_bc_out=RBC_tensor[1:],
            )

            if loss_on_interior:
                loss = criterion(pred[:, 0, 1:-1], gamma_np1[:, 1:-1])
            else:
                loss = criterion(pred[:, 0, :], gamma_np1)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf detected at epoch {epoch}; stopping.")
                break

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip
                )

            optimizer.step()

            if scheduler_obj:
                if isinstance(scheduler_obj, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler_obj.step(loss.item())
                else:
                    scheduler_obj.step()

            # Record loss, alpha, beta, and Ds.
            alpha_value = float(model.alpha.detach().cpu())
            beta_value = float(model.beta.detach().cpu())
            Ds_value = model.get_Ds()

            hist.append([
                loss.item(),
                alpha_value,
                beta_value,
                Ds_value,
            ])

            if print_every_transport and epoch % print_every_transport == 0:
                diff_kernel, adv_kernel = model.get_kernels()

                print(
                    f"Epoch {epoch:6d} | "
                    f"Loss={loss.item():.6e} | "
                    f"Ds={Ds_value:.6e}"
                )
                print("  diffusion:", diff_kernel)
                print("  advection :", adv_kernel)

            if loss.item() < breaktol:
                print(f"Converged at epoch {epoch}, loss={loss.item():.3e}")
                break

        model.cpu()
        model.device = "cpu"

        return model, np.asarray(hist, dtype=np.float64)

    # Handle multicase vs single case data
    if multicase:
        # trainingdata is a tuple: (inputs, targets, bcs)
        inputs_np, targets_np, bcs_np = trainingdata
        inputs = torch.tensor(inputs_np, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1, nx)
        targets = torch.tensor(targets_np, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1, nx)
        bcs = torch.tensor(bcs_np, dtype=torch.float32, device=device)  # (N, 2)

        # Create dataset and dataloader if batch_size specified
        if batch_size is not None:
            dataset = torch.utils.data.TensorDataset(inputs, targets, bcs)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle
            )
        else:
            dataloader = None
    else:
        # Original behavior: trainingdata is 2D array (Nsteps, nx)
        solution = torch.tensor(trainingdata, dtype=torch.float32, device=device)
        Nsteps, nx = solution.shape

        # Prepare training data
        if not multistep or multistep == 1:
            # single-step
            inputs = solution[:-1, :].unsqueeze(1)   # (Nsteps-1, 1, nx)
            targets = solution[1:, :].unsqueeze(1)   # (Nsteps-1, 1, nx)
        else:
            # multi-step rollout
            Npredict = Nsteps - multistep
            inputs = solution[:Npredict, :].unsqueeze(1)
            targets = torch.stack(
                [solution[t + 1: t + 1 + multistep, :].unsqueeze(1)
                 for t in range(Npredict)], dim=0
            )
        dataloader = None

    # ------------------------------------------------------------------ #
    # Pre-assemble multi-channel inputs ONCE before the training loop.    #
    # This matches the behaviour of the old train_burgers_kernel which     #
    # called prepare_training_data_burgers() upfront — no per-epoch cost. #
    # ------------------------------------------------------------------ #
    use_multichannel = (
        not multicase
        and (not multistep or multistep == 1)
        and hasattr(model, 'in_ch')
        and model.in_ch >= 2
    )

    if use_multichannel:
        u_sq = 0.5 * inputs * inputs   # (T-1, 1, nx)

        if model.in_ch >= 3 and forcing_history is not None:
            # Build forcing tensor once
            if callable(forcing_history):
                f_list = [
                    torch.tensor(forcing_history(n), dtype=torch.float32, device=device)
                    .unsqueeze(0).unsqueeze(0)
                    for n in range(inputs.shape[0])
                ]
                f_stack = torch.cat(f_list, dim=0)          # (T-1, 1, nx)
            else:
                forcing_arr = np.asarray(forcing_history, dtype=np.float32)
                f_np = forcing_arr[:inputs.shape[0]]
                if f_np.ndim == 1:
                    f_np = np.tile(f_np, (inputs.shape[0], 1))
                f_stack = torch.tensor(f_np, dtype=torch.float32, device=device).unsqueeze(1)

            inputs = torch.cat([inputs, u_sq, f_stack], dim=1)   # (T-1, 3, nx)

        elif model.in_ch == 2:
            inputs = torch.cat([inputs, u_sq], dim=1)             # (T-1, 2, nx)

    # Compile model for PyTorch 2.0+ (significant speedup)
    if compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("Model compiled successfully")
        except Exception as e:
            print(f"Model compilation failed: {e}")

    # Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if scheduler == "ReduceLROnPlateau":
        scheduler_obj = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=0.5, patience=50)
    elif scheduler == "OneCycleLR" or scheduler == True:
        steps_per_epoch = len(dataloader) if dataloader else 4
        scheduler_obj = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                                      epochs=epochs, steps_per_epoch=steps_per_epoch)
    elif scheduler:
        scheduler_obj = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    else:
        scheduler_obj = None

    criterion = nn.MSELoss()

    # Setup display and plots
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        plt.ion()
        plt.show()

    if conv_stats:
        Nsize = 1 + model.N
        if added_constraints:
            Nsize += len(added_constraints)
        conv_hist = np.zeros([epochs, Nsize])

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        if dataloader:
            # Batch training for multicase
            for batch_inputs, batch_targets, batch_bcs in dataloader:
                optimizer.zero_grad(set_to_none=True)

                predictions = model(batch_inputs)
                data_loss = criterion(predictions, batch_targets)

                cl1, cl2, cl3 = model.constraint_loss(CFL)
                if added_constraints:
                    loss = data_loss + l1 * cl1 + l2 * cl2 + l3 * cl3
                else:
                    loss = data_loss

                # NaN/Inf check
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf detected at epoch {epoch} — stopping.")
                    break

                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            loss = epoch_loss / n_batches

            if scheduler_obj and scheduler_obj.__class__.__name__ != 'OneCycleLR':
                if isinstance(scheduler_obj, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler_obj.step(loss)
                else:
                    scheduler_obj.step()
        else:
            # Single-batch training
            optimizer.zero_grad(set_to_none=True)

            if not multistep or multistep == 1:
                # inputs already has the right channels (pre-assembled above)
                predictions = model(inputs)
            else:
                # rollout k steps
                preds = []
                u = inputs
                for step in range(multistep):
                    u = model(u)
                    preds.append(u)
                predictions = torch.stack(preds, dim=1)  # (Npredict, multistep, 1, nx)

            data_loss = criterion(predictions, targets)

            cl1, cl2, cl3 = model.constraint_loss(CFL)
            if added_constraints:
                loss = data_loss + l1 * cl1 + l2 * cl2 + l3 * cl3
            else:
                loss = data_loss

            if plot:
                print("Epoch ", epoch, "Loss:", loss.item(),
                      "weights:", model.conv.weight[:].cpu().detach().numpy())
                if multistep:
                    ax[0].pcolormesh(predictions[-2 * Nsteps:, 0, :].cpu().detach().numpy())
                    ax[1].pcolormesh(targets[-2 * Nsteps:, 0, :].cpu().detach().numpy())
                else:
                    ax[0].pcolormesh(predictions[:, 0, :].cpu().detach().numpy())
                    ax[1].pcolormesh(targets[:, 0, :].cpu().detach().numpy())
                plt.pause(0.001)
                ax[0].cla()
                ax[1].cla()

            # NaN/Inf check
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf detected at epoch {epoch} — stopping.")
                break

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            if scheduler_obj:
                if isinstance(scheduler_obj, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler_obj.step(loss.item())
                else:
                    scheduler_obj.step()

        if conv_stats:
            a = np.array(model.conv.weight[:].cpu().detach().numpy()[0][0], dtype=np.float64)
            conv_hist[epoch, 0] = loss if isinstance(loss, float) else loss.item()
            conv_hist[epoch, 1:model.N + 1] = a[:]
            if added_constraints:
                conv_hist[epoch, model.N + 1:model.N + 1 + len(added_constraints)] = [
                    cl1.cpu().detach().numpy(),
                    cl2.cpu().detach().numpy(),
                    cl3.cpu().detach().numpy()
                ]

        # Break condition for low error
        loss_value = loss if isinstance(loss, float) else loss.item()
        if loss_value < breaktol:
            break

        # Print loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            w = model.conv.weight[:].cpu().detach().numpy()[0][0]
            j_values = np.arange(-(model.N // 2), (model.N // 2) + 1)

            weights = w.copy() / CFL
            mid = int(w.shape[0] / 2.)
            weights[mid] = (w[mid] - 1.) / CFL

            # # Von Neumann stability
            # Nx = trainingdata.shape[1]
            # m = np.arange(Nx)
            # theta = 2 * np.pi * m / Nx
            # G = np.zeros(Nx, dtype=complex)
            # for jj, ww in zip(j_values, w):
            #     G += ww * np.exp(1j * jj * theta)

            # print("Epoch ", epoch, "/", epochs,
            #       "Loss:", "{:.3e}".format(loss_value),
            #       "weights:", np.round(weights, 8),
            #       "sum w", np.round(w.sum(), 8),
            #       "sum wj", np.round((w * j_values).sum(), 8),
            #       "2CFL-sum wj^2", 2. * CFL - np.round((w * j_values ** 2).sum(), 8),
            #       "Von Neumann G", np.abs(G).max())

            print("Epoch ", epoch, "/", epochs,
                  "Loss:", "{:.3e}".format(loss_value),
                  "weights:", np.round(weights, 8),
                  "sum w", np.round(w.sum(), 8),
                  "sum wj", np.round((w * j_values).sum(), 8),
                  "2CFL-sum wj^2", 2. * CFL - np.round((w * j_values ** 2).sum(), 8))
            if added_constraints:
                print("constraints", cl1.cpu().detach().numpy(),
                      cl2.cpu().detach().numpy(),
                      cl3.cpu().detach().numpy())

    if device:
        model.cpu()
        model.device = "cpu"

    if conv_stats:
        return model, conv_hist[:epoch + 1, :]
    else:
        return model
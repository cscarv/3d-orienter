import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
import trimesh, torch
from torch.utils.data import DataLoader
from dataloaders.MultiMeshFlipperDataset import MultiMeshFlipperDataset
from utils.helpers import zero_out_nan_gradients, is_method, get_log_dir, get_num_workers, visualize_flipper_model_on_mesh
from tqdm import tqdm
import os

class FlipperTrainerModel(pl.LightningModule):
    def __init__(self, 
                specs,
                core_model,
                train_index_file_path,
                val_index_file_path,
                inference_index_file_path,
                preload,
                confusion_matrices,
                num_points_per_cloud,
                train_batch_size,
                val_batch_size,
                unlock_every_k_epochs,
                start_lr
                ):
        super(FlipperTrainerModel, self).__init__()

        self.model = core_model
        self.confusion_matrices = confusion_matrices
        self.up_flipper = up_flipper
        if self.confusion_matrices:
            self.train_loss_fn = nn.NLLLoss()
        else:
            self.train_loss_fn = nn.CrossEntropyLoss()
        self.unlock_every_k_epochs = unlock_every_k_epochs
        self.training_dataloader = DataLoader(MultiMeshFlipperDataset(index_file_path = train_index_file_path, sample_size = num_points_per_cloud, preload=preload, confusion_matrices=confusion_matrices), 
                                                batch_size = train_batch_size,
                                                shuffle = False,
                                                num_workers = get_num_workers(specs["cpus_per_gpu"]), 
                                                persistent_workers = True # else there's overhead on switch
                                                )
        self.validation_dataloader = DataLoader(MultiMeshFlipperDataset(index_file_path = val_index_file_path, sample_size = num_points_per_cloud, preload=preload, confusion_matrices=confusion_matrices), 
                                                batch_size = val_batch_size,
                                                shuffle = False,
                                                num_workers = get_num_workers(specs["cpus_per_gpu"]),
                                                persistent_workers = True # else there's overhead on switch
                                                )
            
        self.inference_meshes = []
        
        # Load the inference meshes from the inference_index_file
        with open(inference_index_file_path, 'r') as f:
            inference_mesh_paths = f.readlines()
        for inference_mesh_path in inference_mesh_paths:
            inference_mesh_path = inference_mesh_path.strip()
            inference_tmesh = trimesh.load(inference_mesh_path, force="mesh")
            inference_verts, inference_faces = self.training_dataloader.dataset.normalize_mesh(inference_tmesh.vertices, inference_tmesh.faces)
            inference_mesh = trimesh.Trimesh(inference_verts, inference_faces)
            self.inference_meshes.append(inference_mesh)

        self.start_lr = start_lr
        self.val_accuracy_sum = 0
        self.val_accuracy_count = 0

    def training_step(self, batch, batch_idx):
        if self.confusion_matrices:
            data_indices, xyz_flipped, normals_flipped, flip_indices, rotation_noise_matrix, confusion_matrix = batch
            # take exp(-confusion_matrix) and row-normalize to get the probabilities
            confusion_matrix = torch.exp(-1*confusion_matrix)
            confusion_matrix = confusion_matrix / confusion_matrix.sum(dim=-1, keepdim=True) # B x 24 x 24
        else:
            data_indices, xyz_flipped, normals_flipped, flip_indices, rotation_noise_matrix = batch
        B, N, D = xyz_flipped.shape
        assert D == 3 # we're assuming 3D points
        # concatenate the xyzs and normals to get a 6D input
        feats_flipped = torch.cat([xyz_flipped, normals_flipped], dim=2)
        logits = self.model(feats_flipped) # B x 24
        if self.confusion_matrices:
            # apply forward correction using the confusion matrix
            # first take softmax of logits to obtain class probabilities
            probs = F.softmax(logits, dim=1).unsqueeze(1) # B x 1 x 24
            # then multiply by the confusion matrix
            confusion_matrix_transp = confusion_matrix.transpose(1, 2) # B x 24 x 24
            smeared_probs = torch.matmul(probs, confusion_matrix_transp).squeeze(1) # B x 24
            # then take the log of the smeared probabilities
            smeared_logits = torch.log(smeared_probs) # B x 24
            # and use this as the input to the loss function
            loss = self.train_loss_fn(smeared_logits, flip_indices)
        else:
            loss = self.train_loss_fn(logits, flip_indices) # cross entropy loss
        self.log('train_loss', loss.item())
            
        return loss

    def validation_step(self, batch, batch_idx):
        # PL only handles multi-gpu/node syncing with log( sync_dist=True), so make sure each batch is the same size or the mean aggregation will be weighted wrong.
        if self.confusion_matrices:
            data_indices, xyz_flipped, normals_flipped, flip_indices, rotation_noise_matrix, confusion_matrix = batch
            # take exp(-confusion_matrix) and row-normalize to get the probabilities
            confusion_matrix = torch.exp(-1*confusion_matrix)
            confusion_matrix = confusion_matrix / confusion_matrix.sum(dim=-1, keepdim=True) # B x 24 x 24
        else:
            data_indices, xyz_flipped, normals_flipped, flip_indices, rotation_noise_matrix = batch
        B, N, D = xyz_flipped.shape
        assert D == 3 # we're assuming 3D points
        # concatenate the xyzs and normals to get a 6D input
        feats_flipped = torch.cat([xyz_flipped, normals_flipped], dim=2)
        logits = self.model(feats_flipped)

        # Compute the accuracy
        if self.confusion_matrices:
            # extract the rows of the confusion matrix corresponding to the correct classes for each batch element
            # confusion_matrix_rows = torch.gather(confusion_matrix, 1, flip_indices.unsqueeze(1).expand(-1, 24)) # B x 24
            # Use advanced indexing to extract the required rows
            rows = torch.arange(B)
            confusion_matrix_rows = confusion_matrix[rows, flip_indices] # B x 24
            # acceptable predictions are any indices in the confusion matrix row that are > 1/24
            # extract list of indices of confusion_mtx_row with entries > 1/24
            acceptable_indices = torch.nonzero(confusion_matrix_rows > 1/24, as_tuple=False) # B x 24
            # construct a list of lists of acceptable indices
            # first entry of each row of acceptable_indices is the list index, and second entry is an element of that list
            # Determine the number of unique lists needed
            num_lists = acceptable_indices[:, 0].max().item() + 1
            # Initialize the list of lists
            acceptable_indices_lists = [[] for _ in range(num_lists)]
            # Iterate over the rows of the tensor and append the second entry to the appropriate list
            for row in acceptable_indices:
                list_index = row[0].item()
                value = row[1].item()
                acceptable_indices_lists[list_index].append(value)
            # take the argmax of the logits
            predicted_flip_indices = torch.argmax(logits, dim=1) # B
            # if predicted_flip_indices is in acceptable_indices, then it's a correct prediction
            correct = 0
            for i in range(B):
                if predicted_flip_indices[i] in acceptable_indices_lists[i]:
                    correct = correct + 1
            self.val_accuracy_sum = self.val_accuracy_sum + correct
            self.val_accuracy_count = self.val_accuracy_count + B
        else:
            predicted_flip_indices = torch.argmax(logits, dim=1)
            correct = (predicted_flip_indices == flip_indices).sum().item()
            self.val_accuracy_sum = self.val_accuracy_sum + correct
            self.val_accuracy_count = self.val_accuracy_count + B

        # Visualize the model's action on each inference mesh and save the results
        # Using a self.current_epoch % 30 == 0 condition doesn't work -- this never evaluates to true

        if batch_idx == 0:
            flip_matrices = self.training_dataloader.dataset.cube_flips
            inference_mesh_dir = f"{get_log_dir(self.logger)}/inference_meshes"
            os.makedirs(inference_mesh_dir, exist_ok=True)
            for i, mesh in enumerate(self.inference_meshes):
                predicted_mesh = visualize_flipper_model_on_mesh(self.model, mesh, flip_matrices)
                predicted_mesh.export(f"{inference_mesh_dir}/epoch_{self.current_epoch}_inference_mesh_{i}.obj")

    def on_after_backward(self):
        zero_out_nan_gradients(self.model)

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            val_accuracy_count = self.all_gather(self.val_accuracy_count)
            val_accuracy_sum = self.all_gather(self.val_accuracy_sum)
            val_accuracy_alldevices = (val_accuracy_sum / val_accuracy_count).item()
            self.log('val_accuracy', val_accuracy_alldevices, sync_dist=False, rank_zero_only = True)
            self.val_accuracy_sum = 0
            self.val_accuracy_count = 0
            tqdm.write(f"{self.current_epoch}, {self.global_step=}, {val_accuracy_alldevices=:.04f}")

    def configure_optimizers(self):
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.start_lr)
        # return optimizer # basic case: no scheduler

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200000, cooldown=0, min_lr=1e-5) # patience was 20
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "frequency": 1,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
                "strict": True,
            },
        }

    def train_dataloader(self):
        return self.training_dataloader
    
    def val_dataloader(self):
        return self.validation_dataloader

    def get_lr(self):
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return sch._last_lr[0]
        else:
            raise Exception('scheduler type not handled')

    def on_train_epoch_end(self):

        # Update selected scheduler is a ReduceLROnPlateau scheduler.
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau) and "train_loss" in self.trainer.callback_metrics:
            sch.step(self.trainer.callback_metrics["train_loss"])
        self.log('lr', self.get_lr())

        # train epoch end is after validation epoch end: https://pytorch-lightning.readthedocs.io/en/1.7.2/common/lightning_module.html#hooks
        if (self.current_epoch + 1) % self.unlock_every_k_epochs == 0:
            if is_method(self.model, 'unlock'):
                self.model.unlock()
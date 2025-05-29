import pytorch_lightning as pl
import trimesh, torch
from torch.utils.data import DataLoader
from dataloaders.MultiMeshDataset import MultiMeshDataset
from utils.losses import chamfer_distance, l2_loss, abs_cos_loss, octahedral_invariant_loss, quotient_regression_loss, nuclear_norm_loss
from utils.helpers import zero_out_nan_gradients, is_method, get_log_dir, get_num_workers, rotation_from_model_outs, visualize_model_on_mesh
from tqdm import tqdm
import os

class OrienterTrainerModel(pl.LightningModule):
    def __init__(self, 
                specs,
                core_model,
                train_loss_fn,
                rotation_representation,
                train_index_file_path,
                val_index_file_path,
                inference_index_file_path,
                preload,
                num_points_per_cloud,
                train_batch_size,
                val_batch_size,
                unlock_every_k_epochs,
                start_lr
                ):
        super(OrienterTrainerModel, self).__init__()

        self.model = core_model
        assert train_loss_fn in ['l2', 'abs_cos', 'octahedral_invariant', 'quotient_regression']
        assert rotation_representation in ["6d", "procrustes"]
        self.train_loss_fn = train_loss_fn
        self.rotation_representation = rotation_representation
        self.val_loss_fn = train_loss_fn # use the same loss function for validation
        self.unlock_every_k_epochs = unlock_every_k_epochs
        self.training_dataloader = DataLoader(MultiMeshDataset(index_file_path = train_index_file_path, sample_size = num_points_per_cloud, preload=preload), 
                                                batch_size = train_batch_size,
                                                shuffle = False,
                                                num_workers = get_num_workers(specs["cpus_per_gpu"]), 
                                                persistent_workers = True # else there's overhead on switch
                                                )
        self.validation_dataloader = DataLoader(MultiMeshDataset(index_file_path = val_index_file_path, sample_size = num_points_per_cloud, preload=preload), 
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
        self.val_chamfer_loss_sum = 0
        self.val_chamfer_loss_count = 0
        if self.train_loss_fn == 'l2' or self.train_loss_fn == 'abs_cos':
            self.val_up_loss_sum = 0
            self.val_up_loss_count = 0
            self.val_front_loss_sum = 0
            self.val_front_loss_count = 0
        elif self.train_loss_fn == 'octahedral_invariant':
            self.val_loss_sum = 0
            self.val_loss_count = 0
        elif self.train_loss_fn == 'quotient_regression':
            self.val_loss_sum = 0
            self.val_loss_count = 0
        elif self.train_loss_fn == 'nuclear_norm':
            self.val_loss_sum = 0
            self.val_loss_count = 0

    def training_step(self, batch, batch_idx):
        data_indices, xyzs_rotated, target_rotation_matrices, normals_rotated = batch
        # randomly rotate the inputs
        B, N, D = xyzs_rotated.shape
        assert D == 3 # we're assuming 3D points
        # concatenate the xyzs and normals to get a 6D input
        feats_rotated = torch.cat([xyzs_rotated, normals_rotated], dim=2)
        if self.rotation_representation == "6d":
            up_predicted, front_predicted = self.model(feats_rotated)
        elif self.rotation_representation == "procrustes":
            predicted_rotations = self.model(feats_rotated) # (B, 3, 3)
            up_predicted = predicted_rotations[:, :, 1] # (B, 3)
            front_predicted = predicted_rotations[:, :, 2] # (B, 3)

        if self.train_loss_fn == 'l2':
            up_loss, front_loss = l2_loss(up_predicted, front_predicted, target_rotation_matrices) # l2 loss returns a tensor of losses for each batch element
            up_loss = up_loss.mean()
            front_loss = front_loss.mean()
            self.log('train_up_loss', up_loss.item())
            self.log('train_front_loss', front_loss.item())
            loss = up_loss + front_loss
        elif self.train_loss_fn == 'abs_cos':
            up_loss, front_loss = abs_cos_loss(up_predicted, front_predicted, target_rotation_matrices) # abs_cos_loss returns a tensor of losses for each batch element
            up_loss = up_loss.mean()
            front_loss = front_loss.mean()
            self.log('train_up_loss', up_loss.item())
            self.log('train_front_loss', front_loss.item())
            loss = up_loss + front_loss
        elif self.train_loss_fn == 'octahedral_invariant':
            loss = octahedral_invariant_loss(up_predicted, front_predicted, target_rotation_matrices) # octahedral_invariant_loss returns a tensor of losses for each batch element
            loss = loss.mean()
            self.log('train_loss', loss.item())
        elif self.train_loss_fn == 'quotient_regression':
            loss = quotient_regression_loss(predicted_rotations, target_rotation_matrices)
            loss = loss.mean()
            self.log('train_loss', loss.item())
        elif self.train_loss_fn == 'nuclear_norm':
            loss = nuclear_norm_loss(predicted_rotations, target_rotation_matrices)
            loss = loss.mean()
            self.log('train_loss', loss.item())
            
        return loss

    def validation_step(self, batch, batch_idx):
        # PL only handles multi-gpu/node syncing with log( sync_dist=True), so make sure each batch is the same size or the mean aggregation will be weighted wrong.
        data_indices, xyzs_rotated, target_rotation_matrices, normals_rotated = batch
        # randomly rotate the inputs
        B, N, D = xyzs_rotated.shape
        assert D == 3 # we're assuming 3D points -- cross product only works in 3D
        # concatenate the xyzs and normals to get a 6D input
        feats_rotated = torch.cat([xyzs_rotated, normals_rotated], dim=2)
        if self.rotation_representation == "6d":
            up_predicted, front_predicted = self.model(feats_rotated)
            predicted_rotations = rotation_from_model_outs(up_predicted, front_predicted)
        elif self.rotation_representation == "procrustes":
            predicted_rotations = self.model(feats_rotated) # (B, 3, 3)
            up_predicted = predicted_rotations[:, :, 1] # (B, 3)
            front_predicted = predicted_rotations[:, :, 2] # (B, 3)

        # apply the predicted and target inverse rotations to the input points
        xyzs_predicted = torch.bmm(xyzs_rotated, predicted_rotations) # B x N x 3
        xyzs_target = torch.bmm(xyzs_rotated, target_rotation_matrices) # B x N x 3

        # loss is chamfer distance between the predicted and target points

        val_chamfer_loss = chamfer_distance(xyzs_predicted, xyzs_target)
        self.val_chamfer_loss_count = self.val_chamfer_loss_count + val_chamfer_loss.shape[0]
        self.val_chamfer_loss_sum = self.val_chamfer_loss_sum + val_chamfer_loss.sum()

        # also track a loss directly on the predicted up- and front-vectors on the validation set

        if self.val_loss_fn == 'l2':
            val_up_loss, val_front_loss = l2_loss(up_predicted, front_predicted, target_rotation_matrices)
            self.val_up_loss_count = self.val_up_loss_count + val_up_loss.shape[0]
            self.val_up_loss_sum = self.val_up_loss_sum + val_up_loss.sum()
            self.val_front_loss_count = self.val_front_loss_count + val_front_loss.shape[0]
            self.val_front_loss_sum = self.val_front_loss_sum + val_front_loss.sum()
        elif self.val_loss_fn == 'abs_cos':
            val_up_loss, val_front_loss = abs_cos_loss(up_predicted, front_predicted, target_rotation_matrices)
            self.val_up_loss_count = self.val_up_loss_count + val_up_loss.shape[0]
            self.val_up_loss_sum = self.val_up_loss_sum + val_up_loss.sum()
            self.val_front_loss_count = self.val_front_loss_count + val_front_loss.shape[0]
            self.val_front_loss_sum = self.val_front_loss_sum + val_front_loss.sum()
        elif self.val_loss_fn == 'octahedral_invariant':
            val_loss = octahedral_invariant_loss(up_predicted, front_predicted, target_rotation_matrices)
            self.val_loss_count = self.val_loss_count + val_loss.shape[0]
            self.val_loss_sum = self.val_loss_sum + val_loss.sum()
        elif self.val_loss_fn == 'quotient_regression':
            val_loss = quotient_regression_loss(predicted_rotations, target_rotation_matrices)
            self.val_loss_count = self.val_loss_count + val_loss.shape[0]
            self.val_loss_sum = self.val_loss_sum + val_loss.sum()
        elif self.val_loss_fn == 'nuclear_norm':
            val_loss = nuclear_norm_loss(predicted_rotations, target_rotation_matrices)
            self.val_loss_count = self.val_loss_count + val_loss.shape[0]
            self.val_loss_sum = self.val_loss_sum + val_loss.sum()

        # Visualize the model's action on each inference mesh and save the results
        # Using a self.current_epoch % 30 == 0 condition doesn't work -- this never evaluates to true

        if batch_idx == 0:
            inference_mesh_dir = f"{get_log_dir(self.logger)}/inference_meshes"
            os.makedirs(inference_mesh_dir, exist_ok=True)
            for i, mesh in enumerate(self.inference_meshes):
                predicted_mesh, predicted_point_cloud = visualize_model_on_mesh(self.model, mesh)
                predicted_mesh.export(f"{inference_mesh_dir}/epoch_{self.current_epoch}_inference_mesh_{i}.obj")
                predicted_point_cloud.export(f"{inference_mesh_dir}/epoch_{self.current_epoch}_inference_point_cloud_{i}.obj")

    def on_after_backward(self):
        zero_out_nan_gradients(self.model)

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero:
            val_chamfer_loss_count = self.all_gather(self.val_chamfer_loss_count)
            val_chamfer_loss_sum = self.all_gather(self.val_chamfer_loss_sum)
            avg_val_chamfer_loss_alldevices = (val_chamfer_loss_sum / val_chamfer_loss_count).item()
            self.log('val_chamfer_loss', avg_val_chamfer_loss_alldevices, sync_dist=False, rank_zero_only = True)
            self.val_chamfer_loss_count = 0
            self.val_chamfer_loss_sum = 0
            tqdm.write(f"{self.current_epoch}, {self.global_step=}, {avg_val_chamfer_loss_alldevices=:.04f}")
            # also log the same function as the training loss on the validation set
            if self.val_loss_fn == 'l2' or self.val_loss_fn == 'abs_cos':
                val_up_loss_count = self.all_gather(self.val_up_loss_count)
                val_up_loss_sum = self.all_gather(self.val_up_loss_sum)
                avg_val_up_loss_alldevices = (val_up_loss_sum / val_up_loss_count).item()
                self.log('val_up_loss', avg_val_up_loss_alldevices, sync_dist=False, rank_zero_only = True)
                self.val_up_loss_count = 0
                self.val_up_loss_sum = 0
                tqdm.write(f"{self.current_epoch}, {self.global_step=}, {avg_val_up_loss_alldevices=:.04f}")
                val_front_loss_count = self.all_gather(self.val_front_loss_count)
                val_front_loss_sum = self.all_gather(self.val_front_loss_sum)
                avg_val_front_loss_alldevices = (val_front_loss_sum / val_front_loss_count).item()
                self.log('val_front_loss', avg_val_front_loss_alldevices, sync_dist=False, rank_zero_only = True)
                self.val_front_loss_count = 0
                self.val_front_loss_sum = 0
                tqdm.write(f"{self.current_epoch}, {self.global_step=}, {avg_val_front_loss_alldevices=:.04f}")
            elif self.val_loss_fn == 'octahedral_invariant' or self.val_loss_fn == 'quotient_regression':
                val_loss_count = self.all_gather(self.val_loss_count)
                val_loss_sum = self.all_gather(self.val_loss_sum)
                avg_val_loss_alldevices = (val_loss_sum / val_loss_count).item()
                self.log('val_loss', avg_val_loss_alldevices, sync_dist=False, rank_zero_only = True)
                self.val_loss_count = 0
                self.val_loss_sum = 0
                tqdm.write(f"{self.current_epoch}, {self.global_step=}, {avg_val_loss_alldevices=:.04f}")

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
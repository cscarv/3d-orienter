from pl_models.OrienterTrainerModel import OrienterTrainerModel
import pytorch_lightning as pl
from utils.helpers import get_timestamp, CustomProgressBar
from ml_models.orienter_model.DGCNNOrienter import DGCNNOrienter
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from pytorch_lightning.loggers import WandbLogger
import argparse 
import os
import json5
import wandb

def main():
    # parse and load specs
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", "-e", default="config/default", help="Path to specs.json5")
    parser.add_argument("--slurm_id", "-s", default=get_timestamp(), help="Path to specs.json5")
    parser.add_argument("--train_index_file", "-ti", default="data/shapenet_index_files/all_point_clouds/train.txt", help="Path to train index file")
    parser.add_argument("--val_index_file", "-vi", default="data/shapenet_index_files/all_point_clouds/val.txt", help="Path to val index file")
    parser.add_argument("--inference_index_file", "-ii", default="data/shapenet_index_files/all_point_clouds/inference.txt", help="Path to inference index file")
    parser.add_argument("--train_loss", "-t", default="quotient_regression", help="Training loss function to use")
    parser.add_argument("--rotation_representation", "-r", default="procrustes", help="Rotation representation to use")
    parser.add_argument("--preload", "-p", action='store_true', help="Preload meshes into memory at initialization")
    parser.add_argument("--resume_from_ckpt", "-rc", default=None, help="Path of checkpoint to resume from")
    parser.add_argument("--resume_from_wandb_run", "-rw", default=None, help="ID of WandB run to resume from")
    args = parser.parse_args()    
    slurm_id = args.slurm_id
    exp_dir = args.exp_dir.rstrip(" /")
    train_index_file = args.train_index_file
    val_index_file = args.val_index_file
    inference_index_file = args.inference_index_file
    train_loss_fn = args.train_loss
    rotation_representation = args.rotation_representation
    preload = args.preload
    resume_from_ckpt = args.resume_from_ckpt
    resume_from_wandb_run = args.resume_from_wandb_run
    assert train_loss_fn in ['l2', 'abs_cos', 'octahedral_invariant', 'quotient_regression', 'nuclear_norm']
    assert rotation_representation in ["6d", "procrustes"]
    # check that rotation_representation is procrustes if train_loss_fn is quotient_regression
    if train_loss_fn == "quotient_regression" or train_loss_fn == "nuclear_norm":
        assert rotation_representation == "procrustes"
    print(f"{exp_dir=}")
    print(f"{slurm_id=}") 
    print(f"{train_index_file=}")
    print(f"{val_index_file=}")
    print(f"{inference_index_file=}")
    print(f"{train_loss_fn=}")
    print(f"{rotation_representation=}")
    print(f"{preload=}")
    print(f"{resume_from_ckpt=}")
    print(f"{resume_from_wandb_run=}")
    with open(os.path.join(exp_dir, "specs.json5"), "r") as file:
        specs = json5.load(file)
    specs["exp_dir"] = exp_dir

    # set up logger
    if resume_from_wandb_run is not None:
        wandb.init(project='orienter-3d', id=resume_from_wandb_run, resume="must")
    else:
        wandb.init(project='orienter-3d', name=f'run_name_{slurm_id}') # can make and set 'dir=' but can't read results there anyway so might as well go to wandb link
    run_id = wandb.run.id
    logger = WandbLogger(save_dir=f'{exp_dir}/{run_id}_{slurm_id}', id=run_id)
    print(f"WandB run URL: {logger.experiment.url}")

    # set up progress bar    
    bar = CustomProgressBar()

    # set up ML model

    dgcnn_args = argparse.Namespace()
    dgcnn_args.k = 20
    dgcnn_args.emb_dims = 1024
    dgcnn_args.dropout = 0.5
    core_model = DGCNNOrienter(dgcnn_args, rotation_representation=rotation_representation)
                          
    # set up pytorch lightning model
    trainer_module = OrienterTrainerModel(
        specs = specs,
        core_model = core_model, 
        train_loss_fn = train_loss_fn,
        rotation_representation = rotation_representation,
        train_index_file_path = train_index_file,
        val_index_file_path = val_index_file,
        inference_index_file_path = inference_index_file,
        preload = preload,
        num_points_per_cloud = 2000,
        train_batch_size = 48,
        val_batch_size = 48,
        unlock_every_k_epochs = 10,
        start_lr = 1e-4
    )

    # set up model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{exp_dir}/{run_id}_{slurm_id}/checkpoints',
        monitor='val_loss',  # Metric to monitor for top_k
        mode='min',  # Mode to monitor for top_k
        filename='checkpoint-{epoch:05d}-{step:05d}-{val_loss:.4f}',  # Naming convention
        save_top_k=3,  # conflicts with saving every_n_epochs 
        save_last=True  # Save the last model state at the end of training
    )

    # set up pytorch lightning training
    trainer = pl.Trainer(max_epochs = 1919,
                        check_val_every_n_epoch=10,
                        num_sanity_val_steps=0,
                        logger=logger,
                        callbacks=[checkpoint_callback, bar],
                        log_every_n_steps=1,
                        precision=32
                        )
    trainer.fit(trainer_module, ckpt_path=resume_from_ckpt)

if __name__ == '__main__':
    main()

import wandb
import os
import time
import torch as th
import torch.utils.data as data_utils
from torchinfo import summary

from src.modules.config import DatasetConfig, SegmentationModelConfig
from src.dataset.HistoDataset import HistoDataset
from src.model.model_wrapper import SegmentationModelWrapper
from src.model.model import SegmentationModel
from src.modules.logger import prepare_folder, get_run_name, bcolors, WandbLogger, save_config
from src.modules.trainer import Trainer

def train(config=None):

    base_log_dir = '/home/cederic/dev/puma/logs/final'

    with wandb.init(
            dir=base_log_dir,
            config=config,
        ):
        config = wandb.config
        base_log_dir = base_log_dir
        if not os.path.exists(base_log_dir):
            os.makedirs(base_log_dir)
        run_name = get_run_name(base_log_dir, f"mode_{config.mode}_lr_{round(config.lr, 5)}_wd_{round(config.weight_decay, 5)}_batch_{config.batch_size}_dropout_{config.dropout}")
        wandb.run.name = run_name  
        wandb.run.save()
        ckpt_save_path, misc_save_path = prepare_folder(base_log_dir, run_name)
        print(f"{bcolors.OKBLUE}Start training with run_name: {bcolors.BOLD}{run_name}{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Log dir: {bcolors.BOLD}{base_log_dir}{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Checkpoint save path: {bcolors.BOLD}{ckpt_save_path}{bcolors.ENDC}")
        print(f"{bcolors.OKBLUE}Misc save path: {bcolors.BOLD}{misc_save_path}{bcolors.ENDC}")
    
        # Configure the model with parameters from the sweep
        train_config = SegmentationModelConfig(
            device=th.device("cuda:4" if th.cuda.is_available() else "cpu"),
            epochs=10000,
            batch_size=config.batch_size,
            split = [0.8, 0.1, 0.1],
            dataset_config=DatasetConfig(
                image_dir="data/01_training_dataset_tif_ROIs",
                geojson_dir_tissue="data/01_training_dataset_geojson_tissue",
                geojson_dir_nuclei="data/01_training_dataset_geojson_nuclei",
                transform=True,
                color_norm=None,
                preprocess=False,
            ),
            mode=config.mode, # tissue or nuclei1 or nuclei2
            feature_extractor_path="/home/cederic/dev/puma/models/ctranspath.pth",
            reset_encoder_parameters=False,
            ckpt_path=None,
            lr=config.lr,
            dropout=config.dropout,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
            T_0=50,
            T_mult=2,
            eta_min=1e-6,
            step_size=10000000,
            gamma=0.1,
            ckpt_save_path=ckpt_save_path,
            misc_save_path=misc_save_path,
            val_every=1000,
            save_max=5,
        )
        save_config(base_log_dir, run_name, train_config.__dict__)
        
        dataset = HistoDataset(**train_config.dataset_config.__dict__)
        dataset_size = len(dataset)
        train_size = int(train_config.split[0] * dataset_size)
        val_size = int(train_config.split[1] * dataset_size)
        test_size = dataset_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = th.utils.data.random_split(dataset, [train_size, val_size, test_size])
        
        train_loader = data_utils.DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        val_loader = data_utils.DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        test_loader = data_utils.DataLoader(
            test_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Initialize model and wrapper
        model = SegmentationModel(config=train_config).to(train_config.device)
        wrapper = SegmentationModelWrapper(model=model, config=train_config, epochs=train_config.epochs)

        summary(model,
           verbose=1,
           input_data={"x": th.rand(1,3,224,224).to(train_config.device)},
        )

        trainer = Trainer(
            device=train_config.device,
            wrapper=wrapper,
            misc_save_path=train_config.misc_save_path,
        )

        trainer.train(
            epochs=train_config.epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            logger=WandbLogger(log_dir=base_log_dir, run_name=run_name),
            ckpt_save_path=train_config.ckpt_save_path,
            ckpt_save_max=train_config.save_max,
            val_every=train_config.val_every,
        )

        trainer.test(
            test_loader=test_loader,
        )


def main_sweep():
    """
    Define the sweep configuration
    """
    
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'val/dice',
            'goal': 'maximize' 
            },
        'parameters': {
            'lr': {
                'values': [1e-5]
            },
            
            'weight_decay': {
                'values': [1e-3]
            },
            
            'batch_size': {
                'values': [16]    
            },
            'dropout': {
                'values': [0.5]
            },
            'mode': {
                'values': ['tissue', 'nuclei1']
            },
        }
    }
    return sweep_config

# run 5 experiments per sweep configuration
if __name__ == "__main__":

    for i in range(1):
        project_name = 'PUMA-SEGMENT'
        # Initialize a sweep
        sweep_config = main_sweep()
        sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
        wandb.agent(sweep_id, function=train, count=2)
        print(f"{bcolors.OKGREEN}Sweep {i} completed!{bcolors.ENDC}")
        time.sleep(4)
    print("All sweeps completed successfully!")

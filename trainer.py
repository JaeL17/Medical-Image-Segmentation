from utils import *
from model import *
import torch.optim as optim
import torch.nn.functional as F
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import os
import argparse
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def run_train(args):
    
    # Seed for to reproduce model
    pl.seed_everything(24, workers=True)
    
    # Intialising custom model
    model = SegmentationModel(
        model_name= args.base_model,
        num_classes= 4,
        init_lr= args.lr,
        optimizer_name=args.optimizer, 
        weight_decay= args.weight_decay, 
        use_scheduler=True, 
        scheduler_name= args. scheduler_name, 
        num_epochs= args.epochs,
    ) 

    data_module = SegmentationDataModule(
    num_classes=DatasetConfig.NUM_CLASSES,
    img_size=DatasetConfig.IMAGE_SIZE,
    ds_mean=DatasetConfig.MEAN,
    ds_std=DatasetConfig.STD,
    batch_size=args.train_batch_size, 
    num_workers=12,
    pin_memory=torch.cuda.is_available(),
    )
    
    # Creating train and validation dataset.
    data_module.prepare_data()
    data_module.setup()
    
    # Creating ModelCheckpoint callback (save based on validation F1 score).
    model_checkpoint = ModelCheckpoint(
    dirpath= os.path.join("./model_output",f"{args.base_model.split('/')[-1]}_trained"),
    monitor="valid/f1",
    mode="max",
    save_top_k = 3,
    save_last = True,
    save_weights_only =True,
    filename="ckpt_{epoch:03d}-vloss_{valid/loss:.4f}_vf1_{valid/f1:.4f}",
    auto_insert_metric_name=False,
    )
    
    # Creating a learning rate monitor callback which will be plotted/added in the default logger
    lr_rate_monitor = LearningRateMonitor(logging_interval="epoch")

    # Initialising the trainer class for model training
    trainer = pl.Trainer(
        accelerator="auto",  # Auto select the best hardware accelerator available
        devices="auto",  # Auto select available devices for the accelerator (For eg. mutiple GPUs)
        strategy="auto",  # Auto select the distributed training strategy.
        max_epochs=args.epochs, # Maximum number of epoch to train for.
        enable_model_summary=False,  # Disable printing of model summary as we are using torchinfo.
        callbacks=[model_checkpoint, lr_rate_monitor],  # Declaring callbacks to use.
        precision=32
    )
 
    # Start training model
    trainer.fit(model, data_module)
    
    # Initialising the trainer class for inference
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,        
        enable_checkpointing=False,
        inference_mode=True,
    )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, required=True)
    parser.add_argument("--weight_decay", type=float, default = 1e-4)
    parser.add_argument("--optimizer", type=str, default = "AdamW")
    parser.add_argument("--scheduler_name", type=str, default="MultiStepLR")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
#     parser.add_argument("--output_model_name", type=str, required=True)
    
    args = parser.parse_args()
    
    run_train(args)
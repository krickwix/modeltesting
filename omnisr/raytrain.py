num_workers = 8
nb_epochs = 1000
batch_size = 64
learning_rate = 1e-4
dataset_enlarge = 10
window_size = 32
data_root="/mnt/repos/ds/DIV2K/"
upsampling = 2

def train_loop_per_worker():
    import ray    
    from ray.train.torch import TorchTrainer
    from ray.train.torch import TorchConfig
    from ray.train import ScalingConfig
    from ray.train import Checkpoint
    from ray.train import RunConfig

    import habana_frameworks.torch.core as htcore
    import torch
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torchvision.transforms import Normalize, ToTensor
    from torchvision import transforms as T
    from torch.utils import data
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2, os, random, shutil
    import components.OmniSR as OmniSR

    class OmniSRDataset(data.Dataset):
        def __init__(self, data_dir="/mnt/ds/DIV2K/", lr_patch_size = 8, image_scale = 2):
            self.hr_image_list = []
            self.lr_image_list = []

            self.data_dir = data_dir

            self.hr_data_dir = os.path.join(data_dir, "DIV2K_train_HR")
            self.lr_data_dir = os.path.join(data_dir, "DIV2K_train_LR_bicubic/X2")

            hr_file_list = sorted(os.listdir(self.hr_data_dir))
            lr_file_list = sorted(os.listdir(self.lr_data_dir))
            for i in range(dataset_enlarge):
                self.hr_image_list.extend(hr_file_list)
                self.lr_image_list.extend(lr_file_list)

            if ray.train.get_context().get_world_rank() == 0:
                print(f"HR Datadir: {self.hr_data_dir}, LR Datadir: {self.lr_data_dir}")
                print(f"Dataset size : {len(self.hr_image_list)}")

            self.i_s            = image_scale
            self.l_ps           = lr_patch_size
            self.h_ps           = lr_patch_size* image_scale

        def __len__(self):
            return len(self.hr_image_list)

        def __getitem__(self, idx):
            hr_img_path = os.path.join(self.hr_data_dir, self.hr_image_list[idx])
            lr_img_path = os.path.join(self.lr_data_dir, self.lr_image_list[idx])

            hr_image      = cv2.imread(hr_img_path)
            hr_image      = cv2.cvtColor(hr_image,cv2.COLOR_BGR2RGB)
            hr_image      = hr_image.transpose((2,0,1))
            hr_image      = torch.from_numpy(hr_image)
            
            lr_image      = cv2.imread(lr_img_path)
            lr_image      = cv2.cvtColor(lr_image,cv2.COLOR_BGR2RGB)
            lr_image      = lr_image.transpose((2,0,1))
            lr_image      = torch.from_numpy(lr_image)

            height   = lr_image.shape[1] # h
            width    = lr_image.shape[2]

            r_h     = random.randint(0,height-self.l_ps)
            r_w     = random.randint(0,width-self.l_ps)
            
            hr_image  = hr_image[:,r_h * self.i_s:(r_h * self.i_s + self.h_ps),
                                    r_w * self.i_s:(r_w * self.i_s + self.h_ps)]
            lr_image  = lr_image[:,r_h:(r_h+self.l_ps),r_w:(r_w+self.l_ps)]

            hr_image = hr_image.float()
            lr_image = lr_image.float()

            return hr_image, lr_image

    def getDataLoader(data_root="data"):
        content_dataset = OmniSRDataset(data_dir=data_root)
        content_dataloader = DataLoader(content_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True
                                        )
        return content_dataloader

    module_params = {
        "upsampling": upsampling,
        "res_num": 5,
        "block_num": 1,
        "bias": True,
        "block_script_name": "OSA",
        "block_class_name": "OSA_Block",
        "window_size": window_size,
        "pe": True,
        "ffn_bias": True
    }
    model = OmniSR.OmniSR(
        3, 3, 64,
        **module_params )
    model = ray.train.torch.prepare_model(model)
    if ray.train.get_context().get_world_rank() == 0:
        print(model)
    dataloader = getDataLoader(data_root="/mnt/repos/ds/DIV2K")
    dataloader = ray.train.torch.prepare_data_loader(dataloader, move_to_device=True, auto_transfer=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  
    criterion = torch.nn.L1Loss()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    checkpoint_dir = "/mnt/repos"

    run_config = RunConfig(storage_path=checkpoint_dir)

    step_epoch  = len(dataloader)
    if ray.train.get_context().get_world_rank() == 0:
        print("Total step = %d in each epoch"%step_epoch)

    for epoch in range(nb_epochs):
        metrics = {}
        for i, (hr, lr) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(lr)
            loss = criterion(output, hr)
            loss.backward()
            htcore.mark_step()
            optimizer.step()
            htcore.mark_step()
            if ray.train.get_context().get_world_rank() == 0:
                print(f"Epoch {epoch} Batch {i} Loss {loss.item()}")
        scheduler.step()
        htcore.mark_step()
        os.makedirs(f"{checkpoint_dir}/Omni-SR/checkpoints/", exist_ok=True)
        torch.save(model.state_dict(), f"{checkpoint_dir}/Omni-SR/checkpoints/model-epoch-{epoch}.pth")
        metrics = {"loss": loss.item()} # Training/validation metrics.
        checkpoint = Checkpoint.from_directory(checkpoint_dir) # Build a Ray Train checkpoint from a directory
        ray.train.report(metrics=metrics) # Report metrics and checkpoint to Ray Train

def main():
    import torch
    from ray.train.torch import TorchTrainer
    from ray.train.torch import TorchConfig
    from ray.train import ScalingConfig
    from ray.train import Checkpoint
    from ray.train import RunConfig
    import ray

    runtime_env = {"pip": ["einops","opencv-python"],
                   "env_vars": {"PT_HPU_ENABLE_LAZY_COLLECTIVES": "true",
                                "PT_HPU_LAZY_MODE": "0"}
    }

    ray.init(runtime_env=runtime_env)
    scaling_config = ScalingConfig(num_workers=num_workers,
                                    resources_per_worker={
                                        "CPU": 1,
                                        "HPU": 1
                                    },
                                   )
    torch_config = TorchConfig(backend = "hccl")
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=scaling_config,
        torch_config=torch_config
    )
    trainer.fit()

if __name__ == '__main__':
    main()
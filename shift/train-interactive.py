"""
Modified SHIFT training script for use with hackathon data. you can change training settings by modifying args in opt
rather than command line

Place this script in the SHIFT/ directory and place the "utils/data.py" file as "hack_data.py" in the root directory
as well.

"""
import time
from models import create_model
from util.visualizer import Visualizer
from torchvision.transforms import Compose, Resize
from torchvision.transforms import InterpolationMode
from hack_data import VIFDataset, train_val_split, make_data_virtual_dir

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


opt = Namespace(
    aug=False,
    batch_size=8,
    beta1=0.5,
    checkpoints_dir="/home/nhp/linux-share/hackathon",
    continue_train=False,
    crop_size=256,
    dataroot="./",
    dataset_mode="aligned",
    direction="AtoB",
    display_env="main",
    display_freq=25,
    display_id=1,
    display_ncols=4,
    display_port=8097,
    display_server="http://localhost",
    display_winsize=256,
    epoch="latest",
    epoch_count=1,
    eps=0.01,
    gan_mode="lsgan",
    gpu_ids=[0],
    init_gain=0.02,
    init_type="normal",
    input_nc=3,
    isTrain=True,
    lambda_A=10.0,
    lambda_B=10.0,
    lambda_L1=100,
    lambda_identity=0.5,
    lambdadapt=False,
    load_iter=0,
    load_size=286,
    lr=0.0002,
    lr_decay_iters=50,
    lr_policy="linear",
    model="cyclegan", # change this to pix2pix or cyclegan
    n_layers_D=3,
    name="experiment_name",
    ndf=64,
    netD="basic",
    netG="resnet_9blocks",
    ngf=64,
    niter=100,
    niter_decay=100,
    no_dropout=True,
    no_flip=False,
    no_html=False,
    norm="instance",
    num_threads=4,
    output_nc=3,
    phase="train",
    pool_size=50,
    preprocess="resize_and_crop",
    print_freq=100,
    save_by_iter=False,
    save_epoch_freq=5,
    save_latest_freq=5000,
    serial_batches=False,
    suffix="",
    thresh_path="./checkpoints",
    tmax=50,
    update_html_freq=1000,
    verbose=False,
)

if __name__ == "__main__":
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(
        opt
    )  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations

    dp = "/home/nhp/linux-share/hackathon/tissue-images"
    samples = ["S04", "S05", "S06"]
    if_dirs = ["AQP1", "Podocalyxin", "Uromodulin"]
    virtual_dir = make_data_virtual_dir(dp, if_dirs=if_dirs, samples=samples)

    tile_data_write_dir = "/home/nhp/linux-share/hackathon/tissue-data/tile-data"
    batch_size = 3
    af_transform = Compose([Resize((256, 256), InterpolationMode.NEAREST), ])
    if_transform = Compose([Resize((256, 256), InterpolationMode.NEAREST), ])

    ## Create the dataset
    ## Normalize whole slide mean scale image to [0, 1] using max dtype value (e.g. for uint16, 2**16-1)
    ## Including adap_reg_thresh makes samples compatible with SHIFT scheme.
    dataset = VIFDataset(
        virtual_dir,
        tile_data_write_dir,
        af_transform=af_transform,
        if_transform=if_transform,
        normalize_whole_slide=True,
        adap_reg_thresh=0.25
    )

    print("Total number samples: ", len(dataset))
    trainloader, valloader = train_val_split(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        train_frac=0.9,
        seed=101,
        shuffle=True,
    )
    dataset_size = len(trainloader) * batch_size
    for epoch in range(
        opt.epoch_count, opt.niter + opt.niter_decay + 1
    ):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        for i, data in enumerate(trainloader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            n_image_in_batch_a = data["A"].shape[0]
            n_image_in_batch_b = data["B"].shape[0]

            data["A_paths"] = ["" for _ in range(n_image_in_batch_b)]
            data["B_paths"] = ["" for _ in range(n_image_in_batch_a)]

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if (
                total_iters % opt.display_freq == 0
            ):  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, save_result
                )

            if (
                total_iters % opt.print_freq == 0
            ):  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t_comp, t_data
                )
                if opt.display_id > 0:
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, losses
                    )

            if (
                total_iters % opt.save_latest_freq == 0
            ):  # cache our latest model every <save_latest_freq> iterations
                print(
                    "saving the latest model (epoch %d, total_iters %d)"
                    % (epoch, total_iters)
                )
                save_suffix = "iter_%d" % total_iters if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if (
            epoch % opt.save_epoch_freq == 0
        ):  # cache our model every <save_epoch_freq> epochs
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, total_iters)
            )
            model.save_networks("latest")
            model.save_networks(epoch)

        print(
            "End of epoch %d / %d \t Time Taken: %d sec"
            % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time)
        )
        model.update_learning_rate()  # update learning rates at the end of every epoch.

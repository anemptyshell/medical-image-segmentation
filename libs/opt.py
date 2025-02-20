import argparse
from libs.utils import str2bool

class options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--epochs", type=int, default=100, metavar='N', help='number of total epochs to run')
        self.parser.add_argument("--batch_size", type=int, default=4)
        self.parser.add_argument("--metric_list", nargs="+", default=["DSC"])   # nargs='+'的作用表示这个参数可以接收多个值，这些值将被组合成一个列表
        ## 
        self.parser.add_argument("--num_workers", type=int, default=0)
        self.parser.add_argument("--seed", type=int, default=1234)
        
        ## 间隔
        self.parser.add_argument("--print_interval", type=int, default=20)
        self.parser.add_argument("--save_interval", type=int, default=10)
        self.parser.add_argument("--val_interval", type=int, default=20)

        self.parser.add_argument("--output", type=str, default="../output")
        self.parser.add_argument("--dataset", default="isic2017", help="Dataset name to use")

        ## optimizer
        self.parser.add_argument('--optimizer_name', default='Adam', choices=['Adam','SGD','AdamW'], help='loss: ' + ' | '.join(['Adam','SGD','AdamW']) + ' (default: Adam)')
        self.parser.add_argument("--weight_decay", type=float, default=0.0001)
        self.parser.add_argument("--momentum", type=float, default=0.9)
        self.parser.add_argument("--lr", default=1e-3, type=float, metavar='LR', help='initial learning rate')
        self.parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')

        ## scheduler
        self.parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
        self.parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
        self.parser.add_argument('--factor', default=0.1, type=float)
        self.parser.add_argument('--patience', default=2, type=int)
        self.parser.add_argument('--milestones', default='1,2', type=str)
        self.parser.add_argument('--gamma', default=2/3, type=float)
        self.parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
        self.parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

        # self.parser.add_argument('--num_workers', default=4, type=int)

        ####
        self.parser.add_argument("--metric", nargs="+", default=["FID"])
        self.parser.add_argument("--phase", type=str, default="train", choices=["train", "test"])

        self.parser.add_argument("--control_print",action="store_true",help="print the results on terminal(default False)")
        self.parser.add_argument("--control_save",action="store_true",help="save the results to files(default False)")
        self.parser.add_argument("--control_save_end",type=int,default=1,help="save the weights on terminal(default False)")


        self.parser.add_argument("--data_path",type=str,default="../data",help="Dataset directory. Please refer Dataset in README.md",)
        self.parser.add_argument("--gpu", default="0", type=str, help="GPU id to use.")
        self.parser.add_argument("--gpu_id", default='0', type=str, help="GPU id to use.")

        self.parser.add_argument("--model_name",type=str,default="unet",help="Prefix of logs and results folders")  ## 日志和结果文件夹的前缀


        self.parser.add_argument("--img_size", default=256, type=int, help="Input image size")
        # self.parser.add_argument("--img_size_H", default=256, type=int, help="Input image size")
        # self.parser.add_argument("--img_size_W", default=256, type=int, help="Input image size")
        self.parser.add_argument("--folds", default=3, type=int, help='define folds number K for K-fold validation')

        # self.parser.add_argument("--weight_decay", type=float, default=0.0001, help="The weight decay")

        self.parser.add_argument("--load_model",default=None, type=str, metavar="PATH", help="path to latest checkpoint (default: None)""ex) --load_model GAN_20190101_101010",)



        self.args = self.parser.parse_known_args()[0]
        self.unknown_args = self.parser.parse_known_args()[1]

        if self.args.phase == "test":
            self.change_args("control_save_end", 0)

    def parse(self):
        self.args = self.parser.parse_known_args()[0]
        self.unknown_args = self.parser.parse_known_args()[1]

    def add_args(self, arg_pairs):
        if not arg_pairs is None:
            for arg_name, arg_value in zip(arg_pairs.keys(), arg_pairs.values()):
                if arg_name in self.unknown_args:
                    if len(self.unknown_args[self.unknown_args.index(arg_name) :]) > 2:
                        if not "--" in self.unknown_args[self.unknown_args.index(arg_name) + 2]:
                            self.parser.add_argument(arg_name, nargs="+", default=arg_value)
                            break
                self.parser.add_argument(arg_name, type=type(arg_value), default=arg_value)
            self.parse()

    def change_args(self, name, value):
        exec('self.parser.set_defaults({} = "{}")'.format(name, value))
        self.parse()

    def get_args(self):
        return self.args

    def get_unknown_args(self):
        return self.unknown_args

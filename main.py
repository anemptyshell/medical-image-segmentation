from libs.opt import options
from libs.data import data

opt = options()

exec("from Med_image_seg.{}.model import {}".format(opt.get_args().model_name, opt.get_args().model_name))

exec("Segment = {}(opt)".format(opt.get_args().model_name))
## Segment = unet(opt)

med_data = data(opt, opt.get_args())

assert opt.unknown_args == [], 'Existing unknown unparsed arguments.'

train_loader, test_loader = med_data.get_loader()


exec("Segment.{}(train_loader, test_loader)".format(opt.get_args().phase))



# ## for dconnnet

# from libs.opt import options

# opt = options()

# exec("from Med_image_seg.{}.model import {}".format(opt.get_args().model_name, opt.get_args().model_name))
# exec("Segment = {}(opt)".format(opt.get_args().model_name))
# # print(opt.unknown_args)
# assert opt.unknown_args == [], 'Existing unknown unparsed arguments.'

# exec("Segment.{}()".format(opt.get_args().phase))

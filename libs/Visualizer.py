import os
import warnings
import cv2
from matplotlib import pyplot as plt
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchextractor as tx
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


class Visualizer:
    def __init__(self, parser):
        super().__init__()
        self.args = parser.get_args()
        self.work_dir = os.path.join(self.args.output, "Med_image_seg", self.args.model_name, self.args.dataset)
        self.log_dir = os.path.join(self.work_dir, "logs")


    def plot_txt(self, file_name, message):
        txt_menu = os.path.join(self.log_dir, file_name)

        with open(txt_menu, "a", newline="\n") as file:
            file.write("%s\n" % message)


    def plot_menu(self, best):
        menu_path = self.log_dir + "/menu.txt"
        menu = os.path.join(menu_path)
        with open(menu, "a", newline="\n") as file:
            datas = [self.args.model_name, best]
            file.write("%s\n" % datas)


    def loggin_metric(self, metric_cluster, current_epoch, best_value, indicator_for_best):
        self.file_name = ""
        self.message = """Epoch : {}, {}'s """.format(current_epoch, self.args.model_name)
        for i in range(len(self.args.metric_list)):
            if not metric_cluster.metric_values_list[i] is None:
                self.message += """{} is {:.4f}. """.format(self.args.metric_list[i], metric_cluster.metric_values_list[i])
                self.message += """ the best {} is {:.4f}; """.format(self.args.metric_list[i], best_value) if i == indicator_for_best else ""

        self.file_name += "metric_epoch{}.txt".format(self.args.epochs)

        self.plot_txt(self.file_name, self.message)
        print(self.message)



    def loggin_loss_lr(self, iter, current_epoch, loss_list, now_lr):
        self.file_name = "train_loss_lr.txt"
        self.message = """Epoch : {}, iter : {}, loss : {:.4f}, lr : {:.6f}""".format(current_epoch, iter, np.mean(loss_list), now_lr)

        self.plot_txt(self.file_name, self.message)
        print(self.message)


    def save_imgs(self, img, gt, pred, iter, save_path, threshold=0.3):

        save_path_img = os.path.join(save_path, 'img')
        save_path_gt = os.path.join(save_path, 'gt')
        save_path_pred = os.path.join(save_path, 'pred')
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        if not os.path.exists(save_path_gt):
            os.makedirs(save_path_gt)
        if not os.path.exists(save_path_pred):
            os.makedirs(save_path_pred)

        size = self.args.img_size / 100

        img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        img = img / 255. if img.max() > 1.1 else img
        gt = np.where(np.squeeze(gt, axis=0) > 0.3, 1, 0)
        # gt = np.copy(np.squeeze(gt, axis=0))
        pred = np.where(np.squeeze(pred, axis=0) > threshold, 1, 0) 
        # pred = np.copy(np.squeeze(pred, axis=0)) 

        # plt.figure(figsize=(2.24,2.24),dpi=100)
        plt.figure(figsize=(size,size),dpi=100)
        # plt.figure(figsize=(5.12,5.12),dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)

        plt.imshow(pred, cmap='gray')  # 如果是灰度图可以指定 cmap='gray'，如果是彩色图无需指定 cmap
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(save_path_pred +'/'+ str(iter) +'.png')
        plt.close()

        # plt.figure(figsize=(2.24,2.24),dpi=100)
        plt.figure(figsize=(size,size),dpi=100)
        # plt.figure(figsize=(5.12,5.12),dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)

        plt.imshow(gt, cmap='gray')  
        plt.axis('off') 
        plt.savefig(save_path_gt +'/'+ str(iter) +'.png')
        plt.close()

        # plt.figure(figsize=(2.24,2.24),dpi=100)
        plt.figure(figsize=(size,size),dpi=100)
        # plt.figure(figsize=(5.12,5.12),dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)

        plt.imshow(img)  
        plt.axis('off') 
        plt.savefig(save_path_img +'/'+ str(iter) +'.png')
        plt.close()



        # plt.figure(figsize=(7,15))

        # plt.subplot(3,1,1)
        # plt.imshow(img)
        # plt.axis('off')

        # plt.subplot(3,1,2)
        # plt.imshow(gt, cmap= 'gray')
        # plt.axis('off')

        # plt.subplot(3,1,3)
        # plt.imshow(pred, cmap = 'gray')
        # plt.axis('off')

        # plt.savefig(save_path + '/' + str(i) +'.png')
        # plt.close()



    def save_img(self, img, gt, pred, iter, save_path):

        save_path_img = os.path.join(save_path, 'img')
        save_path_gt = os.path.join(save_path, 'gt')
        save_path_pred = os.path.join(save_path, 'pred')
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        if not os.path.exists(save_path_gt):
            os.makedirs(save_path_gt)
        if not os.path.exists(save_path_pred):
            os.makedirs(save_path_pred)

        size = self.args.img_size / 100

        pred_array = pred.squeeze(0).squeeze(0).cpu().numpy()
        plt.figure(figsize=(size,size),dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)

        plt.imshow(pred_array, cmap='gray')  # 如果是灰度图可以指定 cmap='gray'，如果是彩色图无需指定 cmap
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(save_path_pred +'/'+ str(iter) +'.png')
        plt.close()


        gt_array = gt.squeeze(0).squeeze(0).cpu().numpy()
        plt.figure(figsize=(size,size),dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)

        plt.imshow(gt_array, cmap='gray')  
        plt.axis('off') 
        plt.savefig(save_path_gt +'/'+ str(iter) +'.png')
        plt.close()

        img_array = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        img_array = img_array / 255. if img_array.max() > 1.1 else img_array
        plt.figure(figsize=(size,size),dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)

        plt.imshow(img_array)  
        plt.axis('off') 
        plt.savefig(save_path_img +'/'+ str(iter) +'.png')
        plt.close()





    def imsave(self, input_image, save_path):
        # 将输入图片矩阵转换成numpy
        if not isinstance(input_image, np.ndarray):
            if isinstance(input_image, torch.Tensor):  # get the data from a variable
                image_tensor = input_image.data
            else:
                return input_image
            image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        else:  # if it is a numpy array, do nothing
            image_numpy = input_image
        im = image_numpy.astype(np.uint8)

        image_pil = Image.fromarray(im)
        image_pil.save(save_path)



    def monitor(self, viz, epoch, lossdic, metric_cluster):  ####在visdom实时监控loss和metric
        """
        visdom 使用
        1.在终端对应环境导入库 pip install visdom
        2.在终端打开visdom服务  python -m visdom.server -p [端口号] （等待几分钟）
        3.在浏览器导航栏输入  http://localhost:8097（部署端口号默认）或http://localhost:端口号
        4.另打开一个终端 运行文件 python main.py

        epoch：当前epoch值
        lossdic：要监控的loss的名称及值,eg.{'loss1':[1,2,3,...],'loss2':[1,2,3,...]}
        metricdic：要监控的metric的名称及值,eg.{'metric1':[1,2,3,...],'metric2':[1,2,3,...]}
        """
        metric_name = self.args.metric
        if self.args.control_monitor:  ###开关控制是否监控
            if epoch == 1:  ####只在第一个epoch的时候新建窗口,保证每次重新运行窗口刷新
                viz.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1, len(lossdic))).cpu(), win="win_loss", opts=dict(xlabel="epoch", ylabel="loss", title="loss", legend=list(lossdic.keys())))  ##,legend=list(lossdic.keys())
                viz.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1, len(metric_name))).cpu(), win="win_metric", opts=dict(xlabel="epoch", ylabel="metric", title="metric", legend=metric_name))  ##,legend=self.metric
            if not viz.win_exists("win_loss"):
                warnings.warn("Created window marked as not existing(win_loss)!")
            for i in range(len(lossdic)):  ######按序追加数据持续监控loss
                viz.line(X=[epoch], Y=[list(lossdic.values())[i][-1]], name=list(lossdic.keys())[i], win="win_loss", update="append")
            if not viz.win_exists("win_metric"):
                warnings.warn("Created window marked as not existing(win_metric)!")
            if metric_cluster is not None:
                for i in range(len(metric_name)):
                    viz.line(X=[epoch], Y=[metric_cluster.get_metric([metric_name[i]])[0]], name=metric_name[i], win="win_metric", update="append")
 


    def save_lossepochimg(self, lossdic, save_pth, pic_name="lossepochimg.jpg"):  ####保存loss-epoch图片
        """
        输入：
            epochs为训练轮次；
            lossname,losslist为输入损失的名称与每个轮次的对应值,
            save_pth,pic_name:为图片保存的路径和名字
        输出：保存图片至对应位置
        """
        epochs = self.args.epochs
        measure = "loss"
        steps_measure = "epochs"
        plt.figure()
        steps = range(1, epochs + 1)
        plt.title("loss-epoch curves")
        ax = plt.gca()
        num_loss = len(lossdic)
        for i in range(num_loss):
            color = next(ax._get_lines.prop_cycler)["color"]
            plt.plot(steps, list(lossdic.values())[i], linewidth=1.5, color=color, linestyle="-", label=list(lossdic.keys())[i])
        plt.xlabel(steps_measure)
        if epochs <= 5:
            plt.xticks(steps)  ####多轮次过于密集
        plt.ylabel(measure)
        plt.legend(loc="best", numpoints=1, fancybox=True)
        save_path = os.path.join(save_pth, "result_img")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, pic_name))
        plt.close()



    def save_metricepochimg(self, metricdic, save_pth, pic_name="metricepochimg.jpg"):  ###保存metric-epoch图片
        """
        输入：
            epochs为训练轮次；
            metricdic为输入指标的名称与每个轮次的对应值
            save_pth,pic_name:为图片保存的路径和名字
        输出：保存图片至对应位置
        """
        epochs = self.args.epochs
        measure = "metric"
        steps_measure = "epochs"
        plt.figure()
        steps = range(1, epochs + 1, self.args.test_interval)
        plt.title("metric-epoch curves")
        ax = plt.gca()
        num_metric = len(metricdic)
        for i in range(num_metric):
            color = next(ax._get_lines.prop_cycler)["color"]
            plt.plot(steps, list(metricdic.values())[i], linewidth=1.5, color=color, linestyle="-", label=list(metricdic.keys())[i])
        plt.xlabel(steps_measure)
        if epochs <= 5:
            plt.xticks(steps)  ####多轮次过于密集
        plt.ylabel(measure)
        plt.legend(loc="best", numpoints=1, fancybox=True)
        save_path = os.path.join(save_pth, "result_img")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, pic_name))
        plt.close()


    def save_attentionmap(self, model_layer_list, save_path, image_size):
        
        """
        model_layer_list: list of dict, like [{'model':self.netd,'layerlist':["DB3.dbblock1","DB1.dbblock2"]},{...}]
            The fully qualified names of the modules producing the relevant feature maps.
        image_path: images you want to test,must put in a file,for example(./testimg).
        save_path: saving path.
        image_size：输入网络的图片大小
        
        Choose the target layer you want to compute the visualization for.
        Usually this will be the last convolutional layer in the model.
        Some common choices can be:
        Resnet18 and 50: model.layer4[-1]
        VGG, densenet161: model.features[-1]
        mnasnet1_0: model.layers[-1]
        You can print the model to help chose the layer
        """
        use_cuda = torch.cuda.is_available()
        num_peer = len(model_layer_list)
        # load img
        imglist = os.listdir(self.args.image_path)
        num_images = len(imglist)

        if self.args.use_normalize:
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.args.mean_map, std=self.args.std_map)])
        else:
            trans = transforms.ToTensor()

        for i in range(num_peer):
            model = model_layer_list[i]["model"]
            layerlist = model_layer_list[i]["layerlist"]
            layer_dict = tx.find_modules_by_names(model, layerlist)
            for name, layer in layer_dict.items():
                cam = GradCAM(model=model, target_layer=layer[-1], use_cuda=use_cuda)
                # init path
                result_path = os.path.join(save_path, "result_img", "attention_result", name)
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                for i in range(num_images):
                    # Load image
                    im_file = os.path.join(self.args.image_path, imglist[i])
                    # input
                    rgb_img = Image.open(im_file, "r")
                    rgb_img = rgb_img.resize((image_size, image_size), Image.ANTIALIAS)
                    rgb_img = np.float32(rgb_img) / 255
                    input_tensor = trans(rgb_img).unsqueeze(0)
                    # Construct the CAM object once, and then re-use it on many images:
                    cam = GradCAM(model=model, target_layer=layer[-1], use_cuda=use_cuda)
                    # If target_category is None, the highest scoring category will be used for every image in the batch.
                    grayscale_cam = cam(input_tensor=input_tensor, target_category=self.args.target_category, aug_smooth=self.args.aug_smooth, eigen_smooth=self.args.eigen_smooth)
                    # In this example grayscale_cam has only one image in the batch:
                    grayscale_cam = grayscale_cam[0, :]
                    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                    # save the original image
                    ori_image = cv2.cvtColor(rgb_img * 255, cv2.COLOR_RGB2BGR)
                    filename = os.path.splitext(imglist[i])[0]
                    filename1 = filename + "_original"
                    self.save_image(ori_image, result_path, filename1)
                    # save heatmap
                    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    filename2 = filename + "_heatmap"
                    self.save_image(heatmap, result_path, filename2)
                    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                    filename3 = filename + "_attented"
                    self.save_image(cam_image, result_path, filename3)


    def save_featuremap(self, model_layer_list, save_pth, image_size, module_filter_fn=None, capture_fn=None):
        """
            Capture the intermediate feature maps of of model.
            Parameters
            ----------
            model: nn.Module,
                The model to extract features from.
            layerlist: list of str, default None
                The fully qualified names of the modules producing the relevant feature maps.
            module_filter_fn: callable, default None
                A filtering function. Takes a module and module name as input and returns True for modules
                producing the relevant features. Either `module_names` or `module_filter_fn` should be
                provided but not both at the same time.
                Example::
                    def module_filter_fn(module, name):
                        return isinstance(module, torch.nn.Conv2d)
                # Hook everything !
                module_filter_fn = lambda module, name: True
                # Capture of all modules inside first layer
                module_filter_fn = lambda module, name: name.startswith("layer1")
                # Focus on all convolutions
                module_filter_fn = lambda module, name: isinstance(module, torch.nn.Conv2d)
            capture_fn: callable, default None
                Operation to carry at each forward pass. The function should comply to the following interface.
                Example::
                    def capture_fn(
                            module: nn.Module,
                            input: Any,
                            output: Any,
                            module_name:str,
                            feature_maps: Dict[str, Any]
                        ):
                        feature_maps[module_name] = output
        
            """
        # get module info
        # print(tx.list_module_names(model))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        imglist = os.listdir(self.args.image_path)
        num_peer = len(model_layer_list)
        num_images = len(imglist)
        # processing
        if self.args.use_normalize:
            trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.args.mean_map, std=self.args.std_map)])
        else:
            trans = transforms.ToTensor()

        for i in range(num_peer):
            model = model_layer_list[i]["model"]
            layerlist = model_layer_list[i]["layerlist"]
            model = tx.Extractor(model, [layer for layer in layerlist], module_filter_fn, capture_fn)

            for j in range(num_images):
                im_file = os.path.join(self.args.image_path, imglist[j])
                filename = os.path.splitext(imglist[j])[0]
                img = Image.open(im_file).convert("RGB")
                img = img.resize((image_size, image_size), Image.ANTIALIAS)
                input_tensor = trans(img).unsqueeze(0).to(device)
                # img_tensor = preprocess_image(img, mean=self.args.mean, std=self.args.std)
                _, features = model(input_tensor)
                therd_size = 256
                for name, f in features.items():
                    features = f[0]
                    iter_range = features.shape[0]
                    for q in range(iter_range):
                        if "fc" in name:
                            continue
                        feature = features.data.cpu().numpy()
                        feature_img = feature[q, :, :]
                        feature_img = np.asarray(feature_img * 255, dtype=np.uint8)

                        dst_path = os.path.join(save_pth, "result_img", "feature_result", filename, name)
                        if not os.path.exists(dst_path):
                            os.makedirs(dst_path)

                        feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
                        if feature_img.shape[0] < therd_size:
                            tmp_file = os.path.join(dst_path, str(q) + "_" + str(therd_size) + ".png")
                            tmp_img = feature_img.copy()
                            tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
                            cv2.imwrite(tmp_file, tmp_img)

                        dst_file = os.path.join(dst_path, str(q) + ".png")
                        cv2.imwrite(dst_file, feature_img)


    def save_filter(self, model_layer_list, save_path):
        """You can get relevant details in a dictionary by calling extractor.info()"""
        num_peer = len(model_layer_list)
        for i in range(num_peer):
            model = model_layer_list[i]["model"]
            layerlist = model_layer_list[i]["layerlist"]
            if not layerlist is not None and set(layerlist).issubset(set(tx.list_module_names(model))):
                warnings.warn("You should either specify the fully qualifying names")
            # layer_dict = tx.find_modules_by_names(model,layerlist)
            parm = {}
            num_layer = len(layerlist)
            for i in range(num_layer):
                for name, parameters in model.named_parameters():
                    if layerlist[i] in name and "weight" in name:
                        parm[name] = parameters
            # Visualising the filters
            for name, parameters in parm.items():
                if parameters.shape[-1] == 1 or len(parameters.shape) == 1:
                    continue
                f_min, f_max = parameters.min(), parameters.max()
                parameters = (parameters - f_min) / (f_max - f_min)
                plt.figure(figsize=(35, 35))
                save_filter_path = os.path.join(save_path, "result_img", "filter_result", name)
                if not os.path.exists(save_filter_path):
                    os.makedirs(save_filter_path)
                # plot first few filters
                for i in range(parameters.shape[0]):
                    # specify subplot and turn of axix
                    if i >= 64:
                        break
                    ax = plt.subplot(8, 8, i + 1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # plot filter channel in grayscale
                    plt.imshow(parameters[i, 0, :, :].data.cpu().numpy(), cmap="gray")  # coolwarm
                    plt.axis("off")

                plt.savefig(save_filter_path + "/filtermap_%s.png" % (name))
                plt.close()



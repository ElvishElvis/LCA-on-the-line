'''
Extract logits for VMs
'''
import sys
import torch
sys.path.append('..')
import torchvision
import torchvision.transforms as transforms
import datasets
import progmet


def extract_logit(dataset_name,image_dataloader,vision_model,label_map):
    # calculate similarity & acc
    logit_list=[]
    meter = progmet.ProgressMeter('apply', interval_time=5)
    for batch_index, minibatch in enumerate(itertools.islice(meter(image_dataloader), None)):
        inputs, gt_labels = minibatch
        image_features=inputs
        logit=vision_model(image_features.cuda()).detach().cpu()
        logit_list.append(logit)

    logit_list_ = torch.concat(logit_list).cpu()
    torch.save(logit_list_, 'llogit/{}'.format(dataset_name))




if __name__ == '__main__':


    dataset_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        
        ])
    # 36 Vision models trained from ImageNet
    # Models were selected randomly and extracted in different batches, thus have some random naming issues.

    # start naming from 0
    resnet50=torchvision.models.resnet50(weights='DEFAULT') # resnet_0
    efficientnet_b0=torchvision.models.efficientnet_b0(weights='DEFAULT') # EfficientNet_1
    densenet201=torchvision.models.densenet201(weights='DEFAULT') # DenseNet_2
    mnasnet1_3=torchvision.models.mnasnet1_3(weights='DEFAULT') # MNASNet_3
    regnet_y_1_6gf=torchvision.models.regnet_y_1_6gf(weights='DEFAULT') # RegNet_4
    resNet101=torchvision.models.resnet101(weights='DEFAULT') # resNet_5
    resNet152=torchvision.models.resnet152(weights='DEFAULT') # resNet_6
    
    # start naming from 5
    swin_b=torchvision.models.swin_b(weights='DEFAULT') # SwinTransformer_5
    inception_v3=torchvision.models.inception_v3(weights='DEFAULT')# Inception3_6
    vit_b_32=torchvision.models.vit_b_32(weights='DEFAULT') # VisionTransformer_7
    vit_l_32=torchvision.models.vit_l_32(weights='DEFAULT') # VisionTransformer_8
    shufflenet_v2_x2_0=torchvision.models.shufflenet_v2_x2_0(weights='DEFAULT') # ShuffleNetV2_9  
    vgg19_bn=torchvision.models.vgg19_bn(weights='DEFAULT') # VGG_10
    wide_resnet101_2=torchvision.models.wide_resnet101_2(weights='DEFAULT')# resNet_11
    convnext_tiny=torchvision.models.convnext_tiny(weights='DEFAULT') #ConvNeXt_12

    # start naming from 10
    alexnet=torchvision.models.alexnet(weights='DEFAULT') # AlexNet_10
                                                        # invalid model, skip 11
    resnet18=torchvision.models.resnet18(weights='DEFAULT') # resNet12
    resnet34=torchvision.models.resnet34(weights='DEFAULT') # resNet13
    vgg11=torchvision.models.vgg11(weights='DEFAULT') # VGG_14
    vgg13=torchvision.models.vgg13(weights='DEFAULT') # VGG_15
    vgg16=torchvision.models.vgg16(weights='DEFAULT') # VGG_16
    vgg19=torchvision.models.vgg19(weights='DEFAULT') # VGG_17
    mnasnet0_5=torchvision.models.mnasnet0_5(weights='DEFAULT') # MNASNet_18
    mnasnet0_75=torchvision.models.mnasnet0_75(weights='DEFAULT') # MNASNet_19
    mnasnet1_0=torchvision.models.mnasnet1_0(weights='DEFAULT') # MNASNet_20
    googlenet=torchvision.models.googlenet(weights='DEFAULT') # GoogLeNet_21
    squeezenet1_0=torchvision.models.squeezenet1_0(weights='DEFAULT') # SqueezeNet_22
    squeezenet1_1=torchvision.models.squeezenet1_1(weights='DEFAULT') # SqueezeNet_23
    vgg11_bn=torchvision.models.vgg11_bn(weights='DEFAULT') # VGG_24
    vgg13_bn=torchvision.models.vgg13_bn(weights='DEFAULT') # VGG_25
    vgg16_bn=torchvision.models.vgg16_bn(weights='DEFAULT') # VGG_26
    densenet121=torchvision.models.densenet121(weights='DEFAULT') # DenseNet_27
    densenet161=torchvision.models.densenet161(weights='DEFAULT') # DenseNet_28
    densenet169=torchvision.models.densenet169(weights='DEFAULT') # DenseNet_29
    mobilenet_v3_small=torchvision.models.mobilenet_v3_small(weights='DEFAULT') # MobileNetV3_30
    mobilenet_v3_large=torchvision.models.mobilenet_v3_large(weights='DEFAULT') # MobileNetV3_31


    model_candidates_list=[resnet18] # add models to this list
    for dataset_name in ['objectnet','imagenet','imagenetv2','imagenet_sketch','imagenet_a','imagenet_r']:
        for _,vision_model in enumerate(model_candidates_list):
            vision_model=torch.nn.DataParallel(vision_model)
            vision_model.eval()
            vision_model.cuda()
            if 'imagenet' in dataset_name:
                test_dataset, labels, label_map = datasets.build_imagenet_dataset(dataset_name, 'test', dataset_transform)
            else:
                test_dataset, labels, _, label_map = datasets.build_objectnet_dataset(dataset_transform)
            image_dataloader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=2)
            file_name_part=f'{dataset_name}_{vision_model.module.__class__.__name__}'
            extract_logit(file_name_part,image_dataloader,vision_model,label_map)




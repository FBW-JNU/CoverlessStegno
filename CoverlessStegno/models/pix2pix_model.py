import torch
import torch.nn.functional as F
import models.networks as networks 
import util.util as util


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        self.alpha = 1

        self.net = torch.nn.ModuleDict(self.initialize_networks(opt)) 

        # set loss functions
        if opt.isTrain:
            self.vggnet_fix = networks.correspondence.VGG19_feature_color_torchversion(vgg_normal_correct=opt.vgg_normal_correct)
            self.vggnet_fix.load_state_dict(torch.load('vgg19_conv.pth'))
            self.vggnet_fix.eval()
            for param in self.vggnet_fix.parameters():
                param.requires_grad = False

            self.vggnet_fix.to(self.opt.gpu_ids[0])
            self.contextual_forward_loss = networks.ContextualLoss_forward(opt)

            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)  
            self.criterionFeat = torch.nn.L1Loss()
            self.MSE_loss = torch.nn.MSELoss()
            if opt.which_perceptual == '5_2': 
                self.perceptual_layer = -1
            elif opt.which_perceptual == '4_2':
                self.perceptual_layer = -2

    def forward(self, data, mode, GforD=None, alpha=1):
       
        input_label, input_semantics, real_image, self_ref, ref_image, ref_label, ref_semantics, secret_image = self.preprocess_input(data, ) 

        self.alpha = alpha
        if mode == 'inference':
            out = {}
            with torch.no_grad():
                out = self.inference(input_semantics,         
                        ref_semantics=ref_semantics, ref_image=ref_image, self_ref=self_ref, secret_image=secret_image)
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            return out
        else:
            raise ValueError("|mode| is invalid")



    def initialize_networks(self, opt):
        net = {}
        net['netG'] = networks.define_G(opt)  
        net['netD'] = networks.define_D(opt) if opt.isTrain else None 
        net['netCorr'] = networks.define_Corr(opt)
        net['netDomainClassifier'] = networks.define_DomainClassifier(opt) if opt.weight_domainC > 0 and opt.domain_rela else None

        if not opt.isTrain or opt.continue_train:
            net['netG'] = util.load_network(net['netG'], 'G', opt.which_epoch, opt)
            if opt.isTrain:
                net['netD'] = util.load_network(net['netD'], 'D', opt.which_epoch, opt)
            net['netCorr'] = util.load_network(net['netCorr'], 'Corr', opt.which_epoch, opt)
            if opt.weight_domainC > 0 and opt.domain_rela:
                net['netDomainClassifier'] = util.load_network(net['netDomainClassifier'], 'DomainClassifier', opt.which_epoch, opt)
            if (not opt.isTrain) and opt.use_ema:
                net['netG'] = util.load_network(net['netG'], 'G_ema', opt.which_epoch, opt)
                net['netCorr'] = util.load_network(net['netCorr'], 'netCorr_ema', opt.which_epoch, opt)
        return net

    def preprocess_input(self, data):
        if self.opt.dataset_mode == 'celebahq':
            glasses = data['label'][:,1::2,:,:].long()
            data['label'] = data['label'][:,::2,:,:]
            glasses_ref = data['label_ref'][:,1::2,:,:].long()
            data['label_ref'] = data['label_ref'][:,::2,:,:]
            if self.use_gpu():
                glasses = glasses.cuda()
                glasses_ref = glasses_ref.cuda()
        elif self.opt.dataset_mode == 'celebahqedge':
            input_semantics = data['label'].clone().cuda().float()
            data['label'] = data['label'][:,:1,:,:]
            ref_semantics = data['label_ref'].clone().cuda().float()
            data['label_ref'] = data['label_ref'][:,:1,:,:]
        elif self.opt.dataset_mode == 'deepfashion':
            input_semantics = data['label'].clone().cuda().float()
            data['label'] = data['label'][:,:3,:,:]
            ref_semantics = data['label_ref'].clone().cuda().float()
            data['label_ref'] = data['label_ref'][:,:3,:,:]

        if self.opt.dataset_mode != 'deepfashion':
            data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['image'] = data['image'].cuda()
            data['ref'] = data['ref'].cuda()
            data['label_ref'] = data['label_ref'].cuda()
            data['image_secret'] = data['image_secret'].cuda()
            if self.opt.dataset_mode != 'deepfashion':
                data['label_ref'] = data['label_ref'].long()
            data['self_ref'] = data['self_ref'].cuda()

        if self.opt.dataset_mode != 'celebahqedge' and self.opt.dataset_mode != 'deepfashion':
            label_map = data['label']
            bs, _, h, w = label_map.size()
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
            input_label = self.FloatTensor(bs, nc, h, w).zero_()
            input_semantics = input_label.scatter_(1, label_map, 1.0)
        
            label_map = data['label_ref']
            label_ref = self.FloatTensor(bs, nc, h, w).zero_()
            ref_semantics = label_ref.scatter_(1, label_map, 1.0)

        if self.opt.dataset_mode == 'celebahq':
            assert input_semantics[:,-3:-2,:,:].sum().cpu().item() == 0
            input_semantics[:,-3:-2,:,:] = glasses
            assert ref_semantics[:,-3:-2,:,:].sum().cpu().item() == 0
            ref_semantics[:,-3:-2,:,:] = glasses_ref
        return data['label'], input_semantics, data['image'], data['self_ref'], data['ref'], data['label_ref'], ref_semantics, data['image_secret']


    def inference(self, input_semantics, ref_semantics=None, ref_image=None, self_ref=None, secret_image=None):
        generate_out = {}
        coor_out = self.net['netCorr'](ref_image, None, input_semantics, ref_semantics, alpha=self.alpha)
        if self.opt.CBN_intype == 'mask':
            CBN_in = input_semantics
        elif self.opt.CBN_intype == 'warp':
            CBN_in = coor_out['warp_out']
        elif self.opt.CBN_intype == 'warp_mask':
            CBN_in = torch.cat((coor_out['warp_out'], input_semantics), dim=1)

        generate_out['fake_image'], generate_out['reveal_image'] = self.net['netG'](input_semantics, warp_out=CBN_in,secret_input=secret_image)
        generate_out = {**generate_out, **coor_out}
        return generate_out

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

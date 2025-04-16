from models.model_utils_pointr import *
from models.model_utils_ours import CrossViewViT
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

class BaseModel_PCN_W3d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.center_num = [512, 128]
        self.encoder_type = 'graph'
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = 3
        self.num_query = query_num = 512
        global_feature_dim = 1024
        # self.view_distance = 1

        ##### Image Encoder #################
        img_layers, in_features = self.get_img_layers(
            'resnet18', feat_size=16)
        self.img_feature_extractor = nn.Sequential(*img_layers)

        self.patch_proj = nn.Linear(128, 256)

        self.increase_dim_img = nn.Sequential(
            nn.Linear(128, 384),
            nn.GELU())

        ##### Point Cloud Encoder #################
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k=16)
        else:
            self.grouper = SimpleEncoder(k=32, embed_dims=512)

        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, 384)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, 384)
        )

        # Coarse Level 1 : Encoder
        self.encoder = PointTransformerEncoderEntry()

        ##### Multi-modality Shape Fusion #################
        self.cross_view_attn = CrossViewViT()

        self.cross_attn1 = cross_attention(384, 384, dropout=0.0, nhead=8)

        ##### Seed Points Generation #################
        self.increase_dim = nn.Sequential(
            nn.Linear(384, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))

        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)
        )

        ##### Point Filtering #################
        self.ranking = nn.Sequential(
            nn.Linear(global_feature_dim+3, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Coarse Level 2 : Decoder
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 128+3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 384)
        )
        self.mem_link = nn.Identity()

        self.decoder = PointTransformerDecoderEntry()

        self.cross_attn2 = cross_attention(384, 384, dropout=0.0, nhead=8)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def get_img_layers(backbone, feat_size):
        """
        Return layers for the image model
        """

        from models.resnet import _resnet, BasicBlock
        assert backbone == 'resnet18'
        layers = [2, 2, 2, 2]
        block = BasicBlock
        backbone_mod = _resnet(
            arch=None,
            block=block,
            layers=layers,
            pretrained=False,
            progress=False,
            feature_size=feat_size,
            zero_init_residual=True)

        all_layers = [x for x in backbone_mod.children()]
        in_features = all_layers[-1].in_features

        # all layers except the final fc layer and the initial conv layers
        # WARNING: this is checked only for resnet models
        main_layers = all_layers[4:-1]
        img_layers = [
            nn.Conv2d(1, feat_size, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(feat_size, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            *main_layers,
            Squeeze()
        ]

        return img_layers, in_features

    def forward(self, xyz, multirgb):
        bs = xyz.size(0)

        ##### Point Cloud Encoding #################
        coor, f = self.grouper(xyz, self.center_num)  # b n c
        pe = self.pos_embed(coor)
        x = self.input_proj(f)

        ##### Image Encoding #################
        multirgb = multirgb.reshape(-1, 1, 224, 224)

        img_feat = self.img_feature_extractor(multirgb).view(bs, 6, -1).contiguous()
        img_feat = self.patch_proj(img_feat)  # [B, 6, 256]

        img_feat_fax = self.cross_view_attn(img_feat)  # [B, 128]

        img_feat = img_feat_fax.unsqueeze(1).expand(-1, x.size(1), -1)

        img_feat = self.increase_dim_img(img_feat)

        ##### Multi-modality Shape Fusion #################
        x = self.cross_attn1(x.permute(0, 2, 1), img_feat.permute(0, 2, 1)).permute(0, 2, 1)

        ##### Seed Points Generation #################
        x = self.encoder(x + pe, coor)  # b n c

        global_feature = self.increase_dim(x)  # B 1024 N
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024

        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)

        ##### Point Filtering (a little different from paper) #################
        corse_feat=torch.cat([global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),coarse], dim=-1)
        confidence_score = self.ranking(corse_feat).reshape(bs, -1, 1)
        idx = torch.argsort(confidence_score, dim=1, descending=True)  # b 512 1
        coarse = torch.gather(coarse, 1, idx[:, :(self.num_query-self.num_query//4)].expand(-1, -1, 3))  # b 384 3

        coarse_inp = misc.fps(xyz, self.num_query // 4)  # B 128 3
        coarse = torch.cat([coarse, coarse_inp], dim=1)  # B 512 3

        ##### Decoder process #################
        x = self.cross_attn2(x.permute(0, 2, 1), img_feat.permute(0, 2, 1)).permute(0, 2, 1)
        mem = self.mem_link(x)

        if self.training:
            picked_points = misc.fps(xyz, 64)
            picked_points = misc.jitter_points(picked_points)
            coarse = torch.cat([coarse, picked_points], dim=1)  # B 512+64 3?
            denoise_length = 64

            # produce query
            q = self.mlp_query(
                torch.cat([
                    global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    img_feat_fax.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    coarse], dim=-1))  # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor, denoise_length=denoise_length)

            return q, coarse, denoise_length, img_feat_fax

        else:
            # produce query
            q = self.mlp_query(
                torch.cat([
                    global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    img_feat_fax.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    coarse], dim=-1))  # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor)

            return q, coarse, 0, img_feat_fax


class BaseModel_PCN_Svd(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.center_num = [512, 128]
        self.encoder_type = 'graph'
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = 3
        self.num_query = query_num = 512
        global_feature_dim = 1024
        # self.view_distance = 1

        ##### Image Encoder #################
        img_layers, in_features = self.get_img_layers(
            'resnet18', feat_size=16)
        self.img_feature_extractor = nn.Sequential(*img_layers)

        self.patch_proj = nn.Linear(128, 256)

        self.increase_dim_img = nn.Sequential(
            nn.Linear(128, 384),
            nn.GELU())

        ##### Point Cloud Encoder #################
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k=16)
        else:
            self.grouper = SimpleEncoder(k=32, embed_dims=512)

        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, 384)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, 384)
        )

        # Coarse Level 1 : Encoder
        self.encoder = PointTransformerEncoderEntry()

        ##### Multi-modality Shape Fusion #################
        self.cross_view_attn = CrossViewViT(init_view='svd')

        self.cross_attn1 = cross_attention(384, 384, dropout=0.0, nhead=8)

        ##### Seed Points Generation #################
        self.increase_dim = nn.Sequential(
            nn.Linear(384, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))

        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)
        )

        ##### Point Filtering #################
        self.ranking = nn.Sequential(
            nn.Linear(global_feature_dim+3, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Coarse Level 2 : Decoder
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 128+3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 384)
        )
        self.mem_link = nn.Identity()

        self.decoder = PointTransformerDecoderEntry()

        self.cross_attn2 = cross_attention(384, 384, dropout=0.0, nhead=8)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def get_img_layers(backbone, feat_size):
        """
        Return layers for the image model
        """

        from models.resnet import _resnet, BasicBlock
        assert backbone == 'resnet18'
        layers = [2, 2, 2, 2]
        block = BasicBlock
        backbone_mod = _resnet(
            arch=None,
            block=block,
            layers=layers,
            pretrained=False,
            progress=False,
            feature_size=feat_size,
            zero_init_residual=True)

        all_layers = [x for x in backbone_mod.children()]
        in_features = all_layers[-1].in_features

        # all layers except the final fc layer and the initial conv layers
        # WARNING: this is checked only for resnet models
        main_layers = all_layers[4:-1]
        img_layers = [
            nn.Conv2d(1, feat_size, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(feat_size, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            *main_layers,
            Squeeze()
        ]

        return img_layers, in_features

    def forward(self, xyz, multirgb):
        bs = xyz.size(0)

        ##### Point Cloud Encoding #################
        coor, f = self.grouper(xyz, self.center_num)  # b n c
        pe = self.pos_embed(coor)
        x = self.input_proj(f)

        ##### Image Encoding #################
        multirgb = multirgb.reshape(-1, 1, 224, 224)

        img_feat = self.img_feature_extractor(multirgb).view(bs, 6, -1).contiguous()
        img_feat = self.patch_proj(img_feat)  # [B, 6, 256]

        img_feat_fax = self.cross_view_attn(img_feat)  # [B, 128]

        img_feat = img_feat_fax.unsqueeze(1).expand(-1, x.size(1), -1)

        img_feat = self.increase_dim_img(img_feat)

        ##### Multi-modality Shape Fusion #################
        x = self.cross_attn1(x.permute(0, 2, 1), img_feat.permute(0, 2, 1)).permute(0, 2, 1)

        ##### Seed Points Generation #################
        x = self.encoder(x + pe, coor)  # b n c

        global_feature = self.increase_dim(x)  # B 1024 N
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024

        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)

        ##### Point Filtering (a little different from paper) #################
        corse_feat=torch.cat([global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),coarse], dim=-1)
        confidence_score = self.ranking(corse_feat).reshape(bs, -1, 1)
        idx = torch.argsort(confidence_score, dim=1, descending=True)  # b 512 1
        coarse = torch.gather(coarse, 1, idx[:, :(self.num_query-self.num_query//4)].expand(-1, -1, 3))  # b 384 3

        coarse_inp = misc.fps(xyz, self.num_query // 4)  # B 128 3
        coarse = torch.cat([coarse, coarse_inp], dim=1)  # B 512 3

        ##### Decoder process #################
        x = self.cross_attn2(x.permute(0, 2, 1), img_feat.permute(0, 2, 1)).permute(0, 2, 1)
        mem = self.mem_link(x)

        if self.training:
            picked_points = misc.fps(xyz, 64)
            picked_points = misc.jitter_points(picked_points)
            coarse = torch.cat([coarse, picked_points], dim=1)  # B 512+64 3?
            denoise_length = 64

            # produce query
            q = self.mlp_query(
                torch.cat([
                    global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    img_feat_fax.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    coarse], dim=-1))  # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor, denoise_length=denoise_length)

            return q, coarse, denoise_length, img_feat_fax

        else:
            # produce query
            q = self.mlp_query(
                torch.cat([
                    global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    img_feat_fax.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    coarse], dim=-1))  # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor)

            return q, coarse, 0, img_feat_fax


class BaseModel_55_W3d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.center_num = [512, 128]
        self.encoder_type = 'graph'
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = 3
        self.num_query = query_num = 512
        global_feature_dim = 1024
        # self.view_distance = 1

        ##### Image Encoder #################
        img_layers, in_features = self.get_img_layers(
            'resnet18', feat_size=16)
        self.img_feature_extractor = nn.Sequential(*img_layers)

        self.patch_proj = nn.Linear(128, 256)

        self.increase_dim_img = nn.Sequential(
            nn.Linear(128, 384),
            nn.GELU())

        ##### Point Cloud Encoder #################
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k=16)
        else:
            self.grouper = SimpleEncoder(k=32, embed_dims=512)

        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, 384)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, 384)
        )

        # Coarse Level 1 : Encoder
        self.encoder = PointTransformerEncoderEntry()

        ##### Multi-modality Shape Fusion #################
        self.cross_view_attn = CrossViewViT(init_view='sn55')

        self.cross_attn1 = cross_attention(384, 384, dropout=0.0, nhead=8)

        ##### Seed Points Generation #################
        self.increase_dim = nn.Sequential(
            nn.Linear(384, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))

        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)
        )

        ##### Point Filtering #################
        self.ranking = nn.Sequential(
            nn.Linear(global_feature_dim+3, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Coarse Level 2 : Decoder
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 128+3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 384)
        )
        self.mem_link = nn.Identity()

        self.decoder = PointTransformerDecoderEntry()

        self.cross_attn2 = cross_attention(384, 384, dropout=0.0, nhead=8)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def get_img_layers(backbone, feat_size):
        """
        Return layers for the image model
        """

        from models.resnet import _resnet, BasicBlock
        assert backbone == 'resnet18'
        layers = [2, 2, 2, 2]
        block = BasicBlock
        backbone_mod = _resnet(
            arch=None,
            block=block,
            layers=layers,
            pretrained=False,
            progress=False,
            feature_size=feat_size,
            zero_init_residual=True)

        all_layers = [x for x in backbone_mod.children()]
        in_features = all_layers[-1].in_features

        # all layers except the final fc layer and the initial conv layers
        # WARNING: this is checked only for resnet models
        main_layers = all_layers[4:-1]
        img_layers = [
            nn.Conv2d(1, feat_size, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(feat_size, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            *main_layers,
            Squeeze()
        ]

        return img_layers, in_features

    def forward(self, xyz, multirgb):
        bs = xyz.size(0)

        ##### Point Cloud Encoding #################
        coor, f = self.grouper(xyz, self.center_num)  # b n c
        pe = self.pos_embed(coor)
        x = self.input_proj(f)

        ##### Image Encoding #################
        multirgb = multirgb.reshape(-1, 1, 224, 224)

        img_feat = self.img_feature_extractor(multirgb).view(bs, 6, -1).contiguous()
        img_feat = self.patch_proj(img_feat)  # [B, 6, 256]

        img_feat_fax = self.cross_view_attn(img_feat)  # [B, 128]

        img_feat = img_feat_fax.unsqueeze(1).expand(-1, x.size(1), -1)

        img_feat = self.increase_dim_img(img_feat)

        ##### Multi-modality Shape Fusion #################
        x = self.cross_attn1(x.permute(0, 2, 1), img_feat.permute(0, 2, 1)).permute(0, 2, 1)

        ##### Seed Points Generation #################
        x = self.encoder(x + pe, coor)  # b n c

        global_feature = self.increase_dim(x)  # B 1024 N
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024

        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)

        ##### Point Filtering (a little different from paper) #################
        corse_feat=torch.cat([global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),coarse], dim=-1)
        confidence_score = self.ranking(corse_feat).reshape(bs, -1, 1)
        idx = torch.argsort(confidence_score, dim=1, descending=True)  # b 512 1
        coarse = torch.gather(coarse, 1, idx[:, :(self.num_query-self.num_query//4)].expand(-1, -1, 3))  # b 384 3

        coarse_inp = misc.fps(xyz, self.num_query // 4)  # B 128 3
        coarse = torch.cat([coarse, coarse_inp], dim=1)  # B 512 3

        ##### Decoder process #################
        x = self.cross_attn2(x.permute(0, 2, 1), img_feat.permute(0, 2, 1)).permute(0, 2, 1)
        mem = self.mem_link(x)

        if self.training:
            picked_points = misc.fps(xyz, 64)
            picked_points = misc.jitter_points(picked_points)
            coarse = torch.cat([coarse, picked_points], dim=1)  # B 512+64 3?
            denoise_length = 64

            # produce query
            q = self.mlp_query(
                torch.cat([
                    global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    img_feat_fax.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    coarse], dim=-1))  # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor, denoise_length=denoise_length)

            return q, coarse, denoise_length, img_feat_fax

        else:
            # produce query
            q = self.mlp_query(
                torch.cat([
                    global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    img_feat_fax.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    coarse], dim=-1))  # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor)

            return q, coarse, 0, img_feat_fax


class BaseModel_55_Svd(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.center_num = [512, 128]
        self.encoder_type = 'graph'
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = 3
        self.num_query = query_num = 512
        global_feature_dim = 1024
        # self.view_distance = 1

        ##### Image Encoder #################
        img_layers, in_features = self.get_img_layers(
            'resnet18', feat_size=16)
        self.img_feature_extractor = nn.Sequential(*img_layers)

        self.patch_proj = nn.Linear(128, 256)

        self.increase_dim_img = nn.Sequential(
            nn.Linear(128, 384),
            nn.GELU())

        ##### Point Cloud Encoder #################
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k=16)
        else:
            self.grouper = SimpleEncoder(k=32, embed_dims=512)

        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, 384)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, 384)
        )

        # Coarse Level 1 : Encoder
        self.encoder = PointTransformerEncoderEntry()

        ##### Multi-modality Shape Fusion #################
        self.cross_view_attn = CrossViewViT(init_view='sn55_svd')

        self.cross_attn1 = cross_attention(384, 384, dropout=0.0, nhead=8)

        ##### Seed Points Generation #################
        self.increase_dim = nn.Sequential(
            nn.Linear(384, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))

        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)
        )

        ##### Point Filtering #################
        self.ranking = nn.Sequential(
            nn.Linear(global_feature_dim+3, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Coarse Level 2 : Decoder
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 128+3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 384)
        )
        self.mem_link = nn.Identity()

        self.decoder = PointTransformerDecoderEntry()

        self.cross_attn2 = cross_attention(384, 384, dropout=0.0, nhead=8)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def get_img_layers(backbone, feat_size):
        """
        Return layers for the image model
        """

        from models.resnet import _resnet, BasicBlock
        assert backbone == 'resnet18'
        layers = [2, 2, 2, 2]
        block = BasicBlock
        backbone_mod = _resnet(
            arch=None,
            block=block,
            layers=layers,
            pretrained=False,
            progress=False,
            feature_size=feat_size,
            zero_init_residual=True)

        all_layers = [x for x in backbone_mod.children()]
        in_features = all_layers[-1].in_features

        # all layers except the final fc layer and the initial conv layers
        # WARNING: this is checked only for resnet models
        main_layers = all_layers[4:-1]
        img_layers = [
            nn.Conv2d(1, feat_size, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(feat_size, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            *main_layers,
            Squeeze()
        ]

        return img_layers, in_features

    def forward(self, xyz, multirgb):
        bs = xyz.size(0)

        ##### Point Cloud Encoding #################
        coor, f = self.grouper(xyz, self.center_num)  # b n c
        pe = self.pos_embed(coor)
        x = self.input_proj(f)

        ##### Image Encoding #################
        multirgb = multirgb.reshape(-1, 1, 224, 224)

        img_feat = self.img_feature_extractor(multirgb).view(bs, 6, -1).contiguous()
        img_feat = self.patch_proj(img_feat)  # [B, 6, 256]

        img_feat_fax = self.cross_view_attn(img_feat)  # [B, 128]

        img_feat = img_feat_fax.unsqueeze(1).expand(-1, x.size(1), -1)

        img_feat = self.increase_dim_img(img_feat)

        ##### Multi-modality Shape Fusion #################
        x = self.cross_attn1(x.permute(0, 2, 1), img_feat.permute(0, 2, 1)).permute(0, 2, 1)

        ##### Seed Points Generation #################
        x = self.encoder(x + pe, coor)  # b n c

        global_feature = self.increase_dim(x)  # B 1024 N
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024

        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)

        ##### Point Filtering (a little different from paper) #################
        corse_feat=torch.cat([global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),coarse], dim=-1)
        confidence_score = self.ranking(corse_feat).reshape(bs, -1, 1)
        idx = torch.argsort(confidence_score, dim=1, descending=True)  # b 512 1
        coarse = torch.gather(coarse, 1, idx[:, :(self.num_query-self.num_query//4)].expand(-1, -1, 3))  # b 384 3

        coarse_inp = misc.fps(xyz, self.num_query // 4)  # B 128 3
        coarse = torch.cat([coarse, coarse_inp], dim=1)  # B 512 3

        ##### Decoder process #################
        x = self.cross_attn2(x.permute(0, 2, 1), img_feat.permute(0, 2, 1)).permute(0, 2, 1)
        mem = self.mem_link(x)

        if self.training:
            picked_points = misc.fps(xyz, 64)
            picked_points = misc.jitter_points(picked_points)
            coarse = torch.cat([coarse, picked_points], dim=1)  # B 512+64 3?
            denoise_length = 64

            # produce query
            q = self.mlp_query(
                torch.cat([
                    global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    img_feat_fax.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    coarse], dim=-1))  # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor, denoise_length=denoise_length)

            return q, coarse, denoise_length, img_feat_fax

        else:
            # produce query
            q = self.mlp_query(
                torch.cat([
                    global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    img_feat_fax.unsqueeze(1).expand(-1, coarse.size(1), -1),
                    coarse], dim=-1))  # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor)

            return q, coarse, 0, img_feat_fax


class PCDreamer_PCN_W3d(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = 384
        self.num_query = 512
        self.num_points = 16384

        self.decoder_type = 'fc'
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'

        self.fold_step = 8
        self.base_model = BaseModel_PCN_W3d(config)

        # NOTE: Final Consolidation is implemented through FoldingNet (a little different from paper)
        if self.decoder_type == 'fold':
            self.factor = self.fold_step ** 2
            self.decode_head = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)
        else:
            if self.num_points is not None:
                self.factor = self.num_points // self.num_query
                assert self.num_points % self.num_query == 0
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2,
                                                        step=self.num_points // self.num_query)
            else:
                self.factor = self.fold_step ** 2
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step ** 2)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.reduce_map = nn.Linear(self.trans_dim + 131, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()
        # self.loss_func = ChamferDistanceHyperV2()

    def get_loss(self, ret, gt):
        pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret

        assert pred_fine.size(1) == gt.size(1)

        # An additional denoise loss is also applied to improve the efficiency and robustness of the model (followed by AdaPoinTr)
        idx = knn_point(self.factor, gt, denoised_coarse)  # B n k
        denoised_target = index_points(gt, idx)  # B n k 3
        denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
        assert denoised_target.size(1) == denoised_fine.size(1)
        loss_denoised = torch.arccosh(1+self.loss_func(denoised_fine, denoised_target))
        loss_denoised = loss_denoised * 0.5

        # recon loss
        # gt_coarse = fps_subsample(gt, pred_coarse.shape[1])
        # loss_coarse = torch.arccosh(1+self.loss_func(pred_coarse, gt_coarse))
        loss_coarse = torch.arccosh(1+self.loss_func(pred_coarse, gt))
        loss_fine = torch.arccosh(1+self.loss_func(pred_fine, gt))
        # loss_coarse = self.loss_func(pred_coarse, gt)
        # loss_fine = self.loss_func(pred_fine, gt)
        loss_recon = loss_coarse + loss_fine

        return loss_denoised, loss_recon, loss_coarse, loss_fine

    def forward(self, xyz, multirgb):
        q, coarse_point_cloud, denoise_length, img_feat_fax = self.base_model(xyz, multirgb)  # B M C and B M 3 576

        B, M, C = q.shape

        rebuild_feature = torch.cat([
            img_feat_fax.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C

        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1))  # BM C
            # rebuild_feature = rebuild_feature.transpose(1, 2).reshape(B * M, -1)
            relative_xyz = self.decode_head(rebuild_feature).reshape(B, M, 3, -1)  # B M 3 S
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2, 3)  # B M S 3

        else:
            rebuild_feature = self.reduce_map(rebuild_feature)  # B M C
            # relative_xyz = self.decode_head(rebuild_feature.transpose(1,2).contiguous())  # B M S 3
            relative_xyz = self.decode_head(rebuild_feature)  # B M S 3
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3

        if self.training:
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()

            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()

            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            ret = (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            return ret

        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query

            ret = (coarse_point_cloud, rebuild_points)
            return ret


class PCDreamer_PCN_Svd(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = 384
        self.num_query = 512
        self.num_points = 16384

        self.decoder_type = 'fc'
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'

        self.fold_step = 8
        self.base_model = BaseModel_PCN_Svd(config)

        # NOTE: Final Consolidation is implemented through FoldingNet (a little different from paper)
        if self.decoder_type == 'fold':
            self.factor = self.fold_step ** 2
            self.decode_head = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)
        else:
            if self.num_points is not None:
                self.factor = self.num_points // self.num_query
                assert self.num_points % self.num_query == 0
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2,
                                                        step=self.num_points // self.num_query)
            else:
                self.factor = self.fold_step ** 2
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step ** 2)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.reduce_map = nn.Linear(self.trans_dim + 131, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()
        # self.loss_func = ChamferDistanceHyperV2()

    def get_loss(self, ret, gt):
        pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret

        assert pred_fine.size(1) == gt.size(1)

        # An additional denoise loss is also applied to improve the efficiency and robustness of the model (followed by AdaPoinTr)
        idx = knn_point(self.factor, gt, denoised_coarse)  # B n k
        denoised_target = index_points(gt, idx)  # B n k 3
        denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
        assert denoised_target.size(1) == denoised_fine.size(1)
        loss_denoised = torch.arccosh(1+self.loss_func(denoised_fine, denoised_target))
        loss_denoised = loss_denoised * 0.5

        # recon loss
        # gt_coarse = fps_subsample(gt, pred_coarse.shape[1])
        # loss_coarse = torch.arccosh(1+self.loss_func(pred_coarse, gt_coarse))
        loss_coarse = torch.arccosh(1+self.loss_func(pred_coarse, gt))
        loss_fine = torch.arccosh(1+self.loss_func(pred_fine, gt))
        # loss_coarse = self.loss_func(pred_coarse, gt)
        # loss_fine = self.loss_func(pred_fine, gt)
        loss_recon = loss_coarse + loss_fine

        return loss_denoised, loss_recon, loss_coarse, loss_fine

    def forward(self, xyz, multirgb):
        q, coarse_point_cloud, denoise_length, img_feat_fax = self.base_model(xyz, multirgb)  # B M C and B M 3 576

        B, M, C = q.shape

        rebuild_feature = torch.cat([
            img_feat_fax.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C

        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1))  # BM C
            # rebuild_feature = rebuild_feature.transpose(1, 2).reshape(B * M, -1)
            relative_xyz = self.decode_head(rebuild_feature).reshape(B, M, 3, -1)  # B M 3 S
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2, 3)  # B M S 3

        else:
            rebuild_feature = self.reduce_map(rebuild_feature)  # B M C
            # relative_xyz = self.decode_head(rebuild_feature.transpose(1,2).contiguous())  # B M S 3
            relative_xyz = self.decode_head(rebuild_feature)  # B M S 3
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3

        if self.training:
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()

            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()

            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            ret = (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            return ret

        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query

            ret = (coarse_point_cloud, rebuild_points)
            return ret


class PCDreamer_55_W3d(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = 384
        self.num_query = 512
        self.num_points = 8192

        self.decoder_type = 'fc'
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'

        self.fold_step = 8
        self.base_model = BaseModel_55_W3d(config)

        # NOTE: Final Consolidation is implemented through FoldingNet (a little different from paper)
        if self.decoder_type == 'fold':
            self.factor = self.fold_step ** 2
            self.decode_head = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)
        else:
            if self.num_points is not None:
                self.factor = self.num_points // self.num_query
                assert self.num_points % self.num_query == 0
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2,
                                                        step=self.num_points // self.num_query)
            else:
                self.factor = self.fold_step ** 2
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step ** 2)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.reduce_map = nn.Linear(self.trans_dim + 131, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        # self.loss_func = ChamferDistanceL1()
        self.loss_func = ChamferDistanceL2()

    def get_loss(self, ret, gt):
        pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret

        assert pred_fine.size(1) == gt.size(1)

        # An additional denoise loss is also applied to improve the efficiency and robustness of the model (followed by AdaPoinTr)
        idx = knn_point(self.factor, gt, denoised_coarse)  # B n k
        denoised_target = index_points(gt, idx)  # B n k 3
        denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
        assert denoised_target.size(1) == denoised_fine.size(1)
        loss_denoised = torch.arccosh(1+self.loss_func(denoised_fine, denoised_target))
        loss_denoised = loss_denoised * 0.5

        # recon loss
        # gt_coarse = fps_subsample(gt, pred_coarse.shape[1])
        # loss_coarse = torch.arccosh(1+self.loss_func(pred_coarse, gt_coarse))
        loss_coarse = torch.arccosh(1+self.loss_func(pred_coarse, gt))
        loss_fine = torch.arccosh(1+self.loss_func(pred_fine, gt))
        # loss_coarse = self.loss_func(pred_coarse, gt)
        # loss_fine = self.loss_func(pred_fine, gt)
        loss_recon = loss_coarse + loss_fine

        return loss_denoised, loss_recon, loss_coarse, loss_fine

    def forward(self, xyz, multirgb):
        q, coarse_point_cloud, denoise_length, img_feat_fax = self.base_model(xyz, multirgb)  # B M C and B M 3 576

        B, M, C = q.shape

        rebuild_feature = torch.cat([
            img_feat_fax.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C

        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1))  # BM C
            # rebuild_feature = rebuild_feature.transpose(1, 2).reshape(B * M, -1)
            relative_xyz = self.decode_head(rebuild_feature).reshape(B, M, 3, -1)  # B M 3 S
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2, 3)  # B M S 3

        else:
            rebuild_feature = self.reduce_map(rebuild_feature)  # B M C
            # relative_xyz = self.decode_head(rebuild_feature.transpose(1,2).contiguous())  # B M S 3
            relative_xyz = self.decode_head(rebuild_feature)  # B M S 3
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3

        if self.training:
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()

            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()

            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            ret = (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            return ret

        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query

            ret = (coarse_point_cloud, rebuild_points)
            return ret


class PCDreamer_55_Svd(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = 384
        self.num_query = 512
        self.num_points = 8192

        self.decoder_type = 'fc'
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'

        self.fold_step = 8
        self.base_model = BaseModel_55_Svd(config)

        # NOTE: Final Consolidation is implemented through FoldingNet (a little different from paper)
        if self.decoder_type == 'fold':
            self.factor = self.fold_step ** 2
            self.decode_head = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)
        else:
            if self.num_points is not None:
                self.factor = self.num_points // self.num_query
                assert self.num_points % self.num_query == 0
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2,
                                                        step=self.num_points // self.num_query)
            else:
                self.factor = self.fold_step ** 2
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step ** 2)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.reduce_map = nn.Linear(self.trans_dim + 131, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        # self.loss_func = ChamferDistanceL1()
        self.loss_func = ChamferDistanceL2()

    def get_loss(self, ret, gt):
        pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret

        assert pred_fine.size(1) == gt.size(1)

        # An additional denoise loss is also applied to improve the efficiency and robustness of the model (followed by AdaPoinTr)
        idx = knn_point(self.factor, gt, denoised_coarse)  # B n k
        denoised_target = index_points(gt, idx)  # B n k 3
        denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
        assert denoised_target.size(1) == denoised_fine.size(1)
        loss_denoised = torch.arccosh(1+self.loss_func(denoised_fine, denoised_target))
        loss_denoised = loss_denoised * 0.5

        # recon loss
        # gt_coarse = fps_subsample(gt, pred_coarse.shape[1])
        # loss_coarse = torch.arccosh(1+self.loss_func(pred_coarse, gt_coarse))
        loss_coarse = torch.arccosh(1+self.loss_func(pred_coarse, gt))
        loss_fine = torch.arccosh(1+self.loss_func(pred_fine, gt))
        # loss_coarse = self.loss_func(pred_coarse, gt)
        # loss_fine = self.loss_func(pred_fine, gt)
        loss_recon = loss_coarse + loss_fine

        return loss_denoised, loss_recon, loss_coarse, loss_fine

    def forward(self, xyz, multirgb):
        q, coarse_point_cloud, denoise_length, img_feat_fax = self.base_model(xyz, multirgb)  # B M C and B M 3 576

        B, M, C = q.shape

        rebuild_feature = torch.cat([
            img_feat_fax.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C

        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1))  # BM C
            # rebuild_feature = rebuild_feature.transpose(1, 2).reshape(B * M, -1)
            relative_xyz = self.decode_head(rebuild_feature).reshape(B, M, 3, -1)  # B M 3 S
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2, 3)  # B M S 3

        else:
            rebuild_feature = self.reduce_map(rebuild_feature)  # B M C
            # relative_xyz = self.decode_head(rebuild_feature.transpose(1,2).contiguous())  # B M S 3
            relative_xyz = self.decode_head(rebuild_feature)  # B M S 3
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3

        if self.training:
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()

            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()

            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            ret = (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            return ret

        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query

            ret = (coarse_point_cloud, rebuild_points)
            return ret
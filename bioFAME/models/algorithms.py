import torch
from torch import nn
import torch.nn.functional as F

from bioFAME.models.transformer_utils import Transformer, FA_Transformer
from bioFAME.models.revin import RevIN

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

ALGORITHMS = [
    'SSL_MLP',
    'bioFAME',
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class SSL_MLP(nn.Module):
    '''single-layer MLP for SSL evaluation'''
    def __init__(self, latent_dim=128, n_classes=5) -> None:
        super().__init__()
        self.mlp = nn.Linear(latent_dim, n_classes)

    def forward(self, l):
        return self.mlp(l)


class bioFAME(nn.Module):
    """
    A Multi-modal transformer-only MAE architecture, free of ConvNets structure
    """
    def __init__(
        self,
        in_channel=1, 
        length=3000,
        n_classes=5,
        hparams={},
        ) -> None:
        super().__init__()

        if hparams['revin']:
            self.revin_layer = RevIN(in_channel, affine=True, subtract_last=False)
        else:
            self.revin_layer = RevIN(in_channel, affine=True, subtract_last=False, skip=True)

        self.hparams = hparams
        patch_size, masking_ratio = hparams['patch_size'], hparams['masking_ratio']
        num_patches = (length // patch_size)
        self.num_patches = num_patches

        self.in_channel = in_channel
        dim = hparams['dim']

        self.encoder_only = hparams['encoder_only']

        # Latent expand/concat procedure
        self.latent_expand = hparams['latent_expand']
        if self.latent_expand == 'cat':
            self.latent_cat = True
        else:
            self.latent_cat = False
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (p d) -> (b c) p d', d=patch_size),
            nn.LayerNorm(patch_size),
            nn.Linear(patch_size, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.mod_embedding = nn.Parameter(torch.randn(1, num_patches*in_channel + 1, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_mode = hparams['cls_token_mode']

        # define FA-Encoder
        self.encoder = FA_Transformer(
            dim=dim, 
            depth=hparams['trans1_depth'], 
            mlp_dim=hparams['trans1_mlpdim'],
            head=hparams['fft_head'], 
            mode=hparams['fft_mode'],
            dropout=0.2,
            post_norm=hparams['post_norm']
            )
        
        # multimodal variants
        if self.latent_cat:
            self.linear_clf = nn.Linear(int(dim*in_channel), n_classes)
            self.linear_clf_dim = int(dim*in_channel)
        else:
            self.linear_clf = nn.Linear(dim, n_classes)
            self.linear_clf_dim = int(dim)

        # MAE section
        self.masking_ratio = masking_ratio
        self.masked_encoder = Transformer(dim=dim, depth=hparams['encoder_depth'], heads=4, dim_head=int(dim/4), mlp_dim=dim, dropout=0.2)
        
        if hparams['modality-decoder']:
            self.masked_decoder = nn.ModuleList([])
            for i in range(in_channel):
                self.masked_decoder.append(
                    Transformer(dim=dim, depth=1, heads=4, dim_head=int(dim/4), mlp_dim=dim, dropout=0.2)
                    )
        else:
            self.masked_decoder = Transformer(dim=dim, depth=1, heads=4, dim_head=int(dim/4), mlp_dim=dim, dropout=0.2)
        
        self.mask_token = nn.Parameter(torch.randn(dim))
        self.decoder_pos_emb = nn.Embedding(num_patches*in_channel, dim)

        self.recon_mod = nn.ModuleList([])
        for _ in range(in_channel):
            self.recon_mod.append(nn.Linear(dim, patch_size))

    def forward_clf(self, x):
        '''forward for classification task'''

        batchxchannel, num_patches, dim = x.shape

        if self.hparams['time_embedding_token']:
            x += self.pos_embedding[:, 1:]

        x = self.encoder(x)
        x = rearrange(x, '(b c) p d -> b (c p) d', c=self.in_channel)

        if self.cls_token_mode == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
            x = torch.cat((cls_tokens, x), dim=1)

        if self.hparams['modality_embedding_token']:
            x += self.mod_embedding[:, :x.shape[1]]
        
        if self.hparams['include_Enc2']:
            x = self.masked_encoder(x)
        else:
            assert self.cls_token_mode != 'cls'

        if self.latent_cat:
            assert self.cls_token_mode == 'avg'
            x = rearrange(x, 'b (c p) d -> b c p d', c=self.in_channel)
            x = torch.mean(x, dim=-2)
            out = rearrange(x, 'b c d -> b (c d)')

        else:
            if self.cls_token_mode == 'avg':
                out = torch.mean(x, dim=1)
            elif self.cls_token_mode == 'cls':
                out = x[:, 0]
            else:
                raise NotImplementedError

        return out

    def forward_pretrain(self, x):
        """reconstruction-based pretraining"""
        batchxchannel, num_patches_p_only, dim = x.shape

        x = self.encoder(x)
        x = rearrange(x, '(b c) p d -> b (c p) d', c=self.in_channel)
        batch, num_patches, dim = x.shape

        num_masked = int(self.masking_ratio * num_patches_p_only)

        # collect rand_indices for future mapback
        rand_indices_collect = []
        rand_indices_mod_collect = []

        masked_indices = []
        unmasked_indices = []

        for channel_i in range(self.in_channel):
            rand_indices_mod = torch.rand(batch, num_patches_p_only, device = x.device).argsort(dim = -1)
            rand_indices_mod_chan = rand_indices_mod + int(channel_i*num_patches_p_only)

            rand_indices_collect.append(rand_indices_mod_chan)
            rand_indices_mod_collect.append(rand_indices_mod)

            masked_indices.append(rand_indices_mod_chan[:, :num_masked])
            unmasked_indices.append(rand_indices_mod_chan[:, num_masked:])

        masked_indices = torch.cat(masked_indices, dim=-1)
        unmasked_indices = torch.cat(unmasked_indices, dim=-1)

        batch_range = torch.arange(batch, device = x.device)[:, None]
        tokens = x[batch_range, unmasked_indices]

        encoded_tokens = self.masked_encoder(tokens) # Encode all separate things together to create mixture
        unmasked_decoder_tokens = encoded_tokens + self.decoder_pos_emb(unmasked_indices)

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked*self.in_channel)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        decoder_tokens = torch.zeros(batch, num_patches, dim, device=x.device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens

        if self.hparams['modality-decoder']:
            decoded_tokens = torch.zeros(batch, num_patches, dim, device=x.device)
            # feed them into different masked decoders
            for channel_i in range(self.in_channel):
                decoded_tokens_mod = self.masked_decoder[channel_i](decoder_tokens)
                decoded_tokens[:, channel_i*num_patches_p_only:(channel_i+1)*num_patches_p_only, :] = decoded_tokens_mod[:, channel_i*num_patches_p_only:(channel_i+1)*num_patches_p_only, :]
        else:
            decoded_tokens = self.masked_decoder(decoder_tokens)

        mask_tokens = decoded_tokens[batch_range, masked_indices]

        return mask_tokens, (batch_range, masked_indices, decoded_tokens, rand_indices_collect, num_masked)

    def forward(self, x, latent_mode=False):
        '''input shape: [batch_size, channel, sequence] --> [batch_size, sequence]'''
        device = x.device

        # RevIN norm
        if self.hparams['revin']:
            x = x.permute(0,2,1)
            x = self.revin_layer(x, 'norm')
            x = x.permute(0,2,1)

        patch = self.to_patch_embedding[0](x)
        patch_emb = self.to_patch_embedding[1:](patch)

        if self.encoder_only:
            """downstream task mode"""
            latents = self.forward_clf(patch_emb)

            if latent_mode:
                return latents
            else:
                return self.linear_clf(latents)
        
        else:
            '''modality specifc training strategy, default for multimodal learning'''
            mask_tokens, (batch_range, masked_indices, decoded_tokens, rand_indices_collect, num_masked) = self.forward_pretrain(patch_emb)            
            new_patch = rearrange(x, 'b c (p d) -> b (c p) d', p=self.num_patches)

            recon_loss = 0.

            collect_reconstruction = []
            collect_original = []

            for channel_i in range(self.in_channel):
                rand_indices_modi = masked_indices[:, int(channel_i*num_masked):int((channel_i+1)*num_masked)]
                original_data_modi = new_patch[batch_range, rand_indices_modi]

                mask_tokens_to_data = self.recon_mod[channel_i](mask_tokens[:, int(channel_i*num_masked):int((channel_i+1)*num_masked)])

                collect_reconstruction.append(mask_tokens_to_data)
                collect_original.append(original_data_modi)

            collect_reconstruction = torch.stack(collect_reconstruction, dim=1)
            collect_original = torch.stack(collect_original, dim=1)

            collect_reconstruction = rearrange(collect_reconstruction, 'b c p d -> b c (p d)')
            collect_original = rearrange(collect_original, 'b c p d -> b c (p d)')
            
            if self.hparams['revin_recover'] and self.hparams['revin']:
                collect_original = collect_original.permute(0,2,1)
                collect_original = self.revin_layer(collect_original, 'denorm')
                collect_original = collect_original.permute(0,2,1)
            
            recon_loss = F.mse_loss(collect_reconstruction, collect_original)
            return recon_loss/self.in_channel
        
    def _mode_adjustment(self, encoder_only=False):
        """adjust model mode"""
        self.encoder_only = encoder_only

    def _channel_adjustment(self, new_channel=1):
        """renew input channels"""
        self.in_channel = new_channel

    def _clf_refresh(self, n_classes):
        """renew clf layer"""
        if self.latent_cat:
            self.linear_clf = nn.Linear(int(self.hparams['dim']*self.in_channel), n_classes)
        else:
            self.linear_clf = nn.Linear(self.hparams['dim'], n_classes)
        
    def _renew_revin(self, new_channel=2):
        """renew revin layer"""
        self.revin_layer._renew_params(new_channel)


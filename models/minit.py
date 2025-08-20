# models/minit.py - Hybrid approach using paper's core with framework adaptations

import torch
import torch.nn.functional as F
import random
from einops import rearrange, repeat
from torch import nn, einsum
from einops.layers.torch import Rearrange

# ============================================================================
# PAPER'S ORIGINAL IMPLEMENTATION (minimal modifications)
# ============================================================================

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., drophead=0):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.attention = nn.Identity()

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.drophead = drophead

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        something = self.attention(attn.clone().detach())

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., drophead=0, layerdrop=0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, drophead=drophead))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
        self.layerdrop = layerdrop
        self.drophead = drophead

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    """Base NeuroImage Transformer - paper's original implementation with framework adaptations"""
    
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, 
                 pool = 'cls', channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0., 
                 drophead=0, layerdrop=0, amp_enabled=False, **kwargs):
        super().__init__()
        
        # Handle both int and tuple inputs (framework adaptation)
        if isinstance(image_size, int):
            image_size = (image_size, image_size, image_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
            
        # For non-cubic volumes, use minimum dimension (framework adaptation)
        if image_size[0] != image_size[1] or image_size[1] != image_size[2]:
            min_dim = min(image_size)
            print(f"Warning: Non-cubic volume {image_size}, using {min_dim}^3 for MINiT")
            image_size = min_dim
            
        # Convert back to single int for paper's implementation
        assert isinstance(image_size, int) or len(set(image_size)) == 1, 'MINiT requires cubic volumes'
        if not isinstance(image_size, int):
            image_size = image_size[0]
            
        assert image_size % patch_size[0] == 0, 'Image dimensions must be divisible by the patch size.'
        assert layerdrop==0 or int(1/layerdrop) == 1/layerdrop, '1/layerdrop needs to be an integer'
        
        num_patches = (image_size // patch_size[0]) ** 3
        patch_dim = channels * (patch_size[0] ** 3)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Use paper's original patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (l p3) -> b (h w l) (p1 p2 p3 c)', 
                     p1 = patch_size[0], p2 = patch_size[0], p3 = patch_size[0]),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, drophead=drophead, layerdrop=layerdrop)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        # Store for framework compatibility
        self.amp_enabled = amp_enabled

    def forward(self, img, mask = None, block_embedding=None):
        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
            x = self.to_patch_embedding(img)
            b, n, _ = x.shape

            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
            
            # Framework adaptation: support block embeddings for MINiT
            if block_embedding is not None:
                x += block_embedding
                
            x = self.dropout(x)
            x = self.transformer(x, mask)

            x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
            x = self.to_latent(x)
            
            return self.mlp_head(x)

class MINiT(nn.Module):
    """Multiple Instance NeuroImage Transformer - paper's original with framework adaptations"""
    
    def __init__(self, *, block_size, image_size, patch_size, num_classes, dim, depth, heads, 
                 mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., 
                 emb_dropout = 0., drophead=0, layerdrop=0, amp_enabled=False, **kwargs):
        super().__init__()
        
        # Framework adaptation: handle different input types
        if isinstance(image_size, (list, tuple)):
            if len(set(image_size)) != 1:
                min_dim = min(image_size)
                print(f"Warning: Non-cubic input {image_size}, using {min_dim}^3")
                image_size = min_dim
            else:
                image_size = image_size[0]
                
        if isinstance(block_size, (list, tuple)):
            block_size = block_size[0]
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
            
        self.image_size = image_size
        self.block_size = block_size
        self.block_count = self.image_size // self.block_size
        self.channels = channels
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.dim = dim
        self.amp_enabled = amp_enabled
        
        # Use paper's original ViT for processing blocks
        self.vit = ViT(
            image_size = block_size,
            patch_size = patch_size, 
            num_classes = num_classes,
            dim = dim, 
            depth = depth, 
            heads = heads, 
            mlp_dim = mlp_dim, 
            pool = pool, 
            channels = channels, 
            dim_head = dim_head, 
            dropout = dropout, 
            emb_dropout = emb_dropout,
            drophead = drophead,
            layerdrop = layerdrop,
            amp_enabled = amp_enabled
        )
        
        self.linear = nn.Linear(self.block_count**3 * self.num_classes, self.num_classes)
        self.block_embeddings = nn.Parameter(torch.randn(self.block_count**3, (block_size//patch_size)**3+1, dim))

    def forward(self, img):
        # Framework adaptation: handle different input sizes
        if img.shape[-3:] != (self.image_size, self.image_size, self.image_size):
            # Resize to expected size
            from torch.nn.functional import interpolate
            img = interpolate(img, size=(self.image_size, self.image_size, self.image_size), 
                            mode='trilinear', align_corners=False)
        
        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
            b = img.shape[0]
            p = self.block_size
            block_count = self.block_count
            
            # Paper's original block processing
            x = rearrange(img, 'b c (h p1) (w p2) (l p3) -> (b h w l) c p1 p2 p3', p1 = p, p2 = p, p3=p)
            results = self.vit(x, block_embedding=self.block_embeddings.repeat(b, 1, 1)).float()
            results = rearrange(results, '(b h w l) n ->  b (h w l n)', h = block_count, w = block_count, l=block_count, n = self.num_classes)
            logits = self.linear(results)

        return logits

# ============================================================================
# FRAMEWORK INTEGRATION LAYER
# ============================================================================

class MiGNiT(nn.Module):
    """Multiple Instance Global NiT - new addition for global attention"""
    
    def __init__(self, *, block_size, image_size, patch_size, num_classes, dim, depth, heads, 
                 mlp_dim, pool = 'cls', channels = 1, dim_head = 64, dropout = 0., 
                 emb_dropout = 0., drophead=0, layerdrop=0, amp_enabled=False, **kwargs):
        super().__init__()
        
        # Same input handling as MINiT
        if isinstance(image_size, (list, tuple)):
            if len(set(image_size)) != 1:
                min_dim = min(image_size)
                print(f"Warning: Non-cubic input {image_size}, using {min_dim}^3")
                image_size = min_dim
            else:
                image_size = image_size[0]
                
        self.image_size = image_size
        self.block_size = block_size if isinstance(block_size, int) else block_size[0]
        self.block_count = self.image_size // self.block_size
        self.num_classes = num_classes
        self.dim = dim
        self.amp_enabled = amp_enabled
        
        # Block-level ViT (outputs embeddings, not classes)
        self.block_vit = ViT(
            image_size = self.block_size,
            patch_size = patch_size, 
            num_classes = dim,  # Output embeddings
            dim = dim, 
            depth = depth, 
            heads = heads, 
            mlp_dim = mlp_dim, 
            pool = pool, 
            channels = channels, 
            dim_head = dim_head, 
            dropout = dropout, 
            emb_dropout = emb_dropout,
            amp_enabled = amp_enabled
        )
        
        # Block embeddings
        patches_per_block = (self.block_size // (patch_size if isinstance(patch_size, int) else patch_size[0])) ** 3
        self.block_embeddings = nn.Parameter(torch.randn(self.block_count**3, patches_per_block + 1, dim))
        
        # Global attention across blocks
        self.global_transformer = Transformer(dim, 2, heads, dim_head, mlp_dim, dropout)
        self.global_cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.final_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        if img.shape[-3:] != (self.image_size, self.image_size, self.image_size):
            from torch.nn.functional import interpolate
            img = interpolate(img, size=(self.image_size, self.image_size, self.image_size), 
                            mode='trilinear', align_corners=False)
        
        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
            b = img.shape[0]
            p = self.block_size
            block_count = self.block_count
            
            # Process blocks to get embeddings
            x = rearrange(img, 'b c (h p1) (w p2) (l p3) -> (b h w l) c p1 p2 p3', p1 = p, p2 = p, p3=p)
            block_embeddings = self.block_vit(x, block_embedding=self.block_embeddings.repeat(b, 1, 1))
            
            # Reshape for global attention
            block_embeddings = rearrange(block_embeddings, '(b h w l) d -> b (h w l) d', 
                                       h = block_count, w = block_count, l = block_count)
            
            # Add global class token
            global_cls_tokens = repeat(self.global_cls_token, '() n d -> b n d', b = b)
            global_input = torch.cat((global_cls_tokens, block_embeddings), dim=1)
            
            # Global attention
            global_output = self.global_transformer(global_input)
            final_embedding = global_output[:, 0]  # Use global class token
            
            return self.final_head(final_embedding)

# ============================================================================
# FACTORY FUNCTIONS FOR FRAMEWORK INTEGRATION
# ============================================================================

def create_nit_model(image_size=64, patch_size=8, num_classes=2, dim=256, depth=6, 
                     heads=8, mlp_dim=1024, dropout=0.1, amp_enabled=False, **kwargs):
    """Create base NeuroImage Transformer using paper's implementation."""
    return ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        amp_enabled=amp_enabled,
        **kwargs
    )

def create_minit_model(image_size=64, block_size=16, patch_size=4, num_classes=2, 
                       dim=256, depth=6, heads=8, mlp_dim=None, dropout=0.1, 
                       amp_enabled=False, **kwargs):
    
    if mlp_dim is None:
        mlp_dim = dim * 4
    """Create MINiT using paper's implementation."""
    return MINiT(
        image_size=image_size,
        block_size=block_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        amp_enabled=amp_enabled,
        **kwargs
    )

def create_mignit_model(image_size=64, block_size=16, patch_size=4, num_classes=2, 
                        dim=256, depth=6, heads=8, mlp_dim=1024, dropout=0.1, 
                        amp_enabled=False, **kwargs):
    """Create MiGNiT with global attention."""
    return MiGNiT(
        image_size=image_size,
        block_size=block_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        amp_enabled=amp_enabled,
        **kwargs
    )

if __name__ == '__main__':
    # Test with framework's typical input size
    print("Testing MINiT with framework integration...")
    
    # Paper's original test
    net = MINiT(
        block_size = 16,
        image_size = 64,
        patch_size = 4,
        num_classes = 2,
        channels = 1,
        dim = 256,
        depth = 6,
        heads = 8,
        mlp_dim = 309
    )
    
    test = torch.ones(2, 1, 64, 64, 64)
    preds = net(test)
    print(f"Paper test passed: {preds.shape}")
    
    # Framework integration test with non-cubic input
    net_framework = create_minit_model(
        image_size=(182, 218, 182),  # Your framework's size
        block_size=16,
        patch_size=4,
        num_classes=2,
        dim=256
    )
    
    test_framework = torch.ones(1, 1, 182, 218, 182)
    preds_framework = net_framework(test_framework)
    print(f"Framework test passed: {preds_framework.shape}")

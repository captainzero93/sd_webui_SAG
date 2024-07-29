Fixed as of 29/07/2024 main branch AUTOMATIC1111 

- Added missing 'process' method - now SAG params show up in PNG info
- Overhauled callback system - no more duplicate callbacks!
- Improved attention module handling for different SD versions
- Added dynamic attention targeting - adapts as sampling progresses
- Beefed up error handling and added some helpful logging
- Updated infotext fields for better UI-parameter mapping
- Cleaned up global variable management
- Made it play nice with SDXL
- Added smart parameter scaling based on image size
- Fixed tensor shape mismatches in cfg_after_cfg_callback

# Implementation of Self Attention Guidance in webui
https://arxiv.org/abs/2210.00939

Demos:
![xyz_grid-0014-232592377.png](resources%2Fimg%2Fxyz_grid-0014-232592377.png)
![xyz_grid-0001-232592377.png](resources%2Fimg%2Fxyz_grid-0001-232592377.png)

For SDXL model, adjust Blur Sigma around 0.34 to prevent blurry image    
Bilinear interpolation: Attention mask is interpolated using bilinear method, resulting sharper image    
Attention target: Choose the block Attention mask would apply to, `dynamic` means depending on noise sigma value    
Base resolution: Change attention mask resolution

# Implicit Slice Agnostic Segmentation Model
Medical image segmentation is increasingly and successfully improved by the use of machine learning algorithms. Despite considerable success, most models are focused on pixel-wise classification of the tissue instead of considering the region of interest as a 3D shape. This work introduces an implicit slice agnostic segmentation (ISAS) model that enables coordinate-based segmentation in a continuous 3D space instead of pixel-wise classification. This is enabled by a continuous feature encoder that can generate features even for regions that are not explicitly mapped in the input CT images. The encoder allows the region of interest to be segmented at almost any resolution and with a flexible number and location of CT images. The model is compared to standard 2D and 3D UNet models for the task of lung segmentation. The experiments reveal that the costs for the additional flexibility are minor cutbacks in segmentation quality in the case of equidistant dense CT slice inputs. In addition, the flexibility of the model in segmenting non-equidistant and sparse CT slice inputs is demonstrated.

# Results

## Benchmarking
### All Models
![](figures/Gifs/all_models.gif)
### Main Models
![](figures/Gifs/main_models.gif)
### Main Models (2D)
![](figures/Gifs/main_model_2d.gif)

## Reconstructions
### Non-ARDS
![](figures/interpol6.gif)
### ARDS
![](figures/interpol284.gif)
### Interpolation 2D UNet vs. ISAS
![](figures/isas_unet_compl.gif)

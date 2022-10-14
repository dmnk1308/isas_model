# Implicit Slice Agnostic Segmentation Model
Medical image segmentation is increasingly and successfully improved by the use of machine learning algorithms. Despite considerable success, most models are focused on pixel-wise classification of the tissue instead of considering the region of interest as a 3D shape. This work introduces an implicit slice agnostic segmentation (ISAS) model that enables coordinate-based segmentation in a continuous 3D space instead of pixel-wise classification. This is enabled by a continuous feature encoder that can generate features even for regions that are not explicitly mapped in the input CT images. The encoder allows the region of interest to be segmented at almost any resolution and with a flexible number and location of CT images. The model is compared to standard 2D and 3D UNet models for the task of lung segmentation. The experiments reveal that the costs for the additional flexibility are minor cutbacks in segmentation quality in the case of equidistant dense CT slice inputs. In addition, the flexibility of the model in segmenting non-equidistant and sparse CT slice inputs is demonstrated.

# Results
## Trainable Features
![](figures/Gifs/trainable_feat.gif)  

## Benchmarking
__Main Models (3D)__  
![](figures/Gifs/main_models.gif)  
__Main Models (2D)__   
![](figures/Gifs/main_model_2d.gif)  
__All Models__  
![](figures/Gifs/all_models.gif)  

## Reconstructions
__Non-ARDS lung__  
![](figures/Gifs/interpol_6.gif)  
__ARDS lung__  
![](figures/Gifs/interpol_284.gif)  
__Interpolation 2D UNet vs. ISAS (Input: 5 CT slices)__  
![](figures/Gifs/isas_unet_compl.gif)  

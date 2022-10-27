# Implicit Slice Agnostic Segmentation Model
Medical image segmentation is increasingly and successfully improved by the use of machine learning algorithms. Despite considerable success, most models are focused on pixel-wise classification of the tissue instead of considering the region of interest as a coherent 3D shape. This work introduces an implicit slice agnostic segmentation (ISAS) model that allows coordinate-based segmentation in a continuous 3D space instead of pixel-wise classification. This is enabled via an encoder-decoder architecture with an encoder that is able to generate features for each coordinate in a continuous 3D space. The decoder performs a classification of given input coordinates by conditioning on a specific object utilizing the encoded features. Given the shape-specific features, the whole shape to be segmented can thus be implicitly represented by the decoder. This allows to output the region of interest at almost any resolution. In addition, the number as well as the distance between individual input slices is variable, which allows a shape reconstruction with sparse input slices as well as an extrapolation in unknown regions of the shape. The model is compared to standard 2D and 3D UNet models for the task of lung segmentation. The experiments reveal that the costs for the additional flexibility are minor cutbacks in segmentation quality in the case of equidistant dense CT slice inputs. In addition, the flexibility of the model in segmenting non-equidistant and sparse CT slice inputs is demonstrated. Based on the results, the architecture of the ISAS model might also be suitable for other input data different from CT images which are not equidistant or whose information content is limited in some areas along the domain that is considered.

# Architecture
__Encoder__<br>
<img src="figures/Architecture/Encoder.jpg" />

__Blocks__<br>
<img src="figures/Architecture/Blocks.jpg" width="600" />

__Decoder__<br>
<img src="figures/Architecture/Decoder.jpg" width="400" />


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
## Morphing
__PIP Values__<br>
<img src="figures/Gifs/peep_interpol.gif" width="400" />

__Different Lungs__<br>
<img src="figures/Gifs/morph.gif" width="400" />




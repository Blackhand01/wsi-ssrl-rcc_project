**RCC Whole histological Slide Images (WSIs) dataset**  
	  
**\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!**  
**Disclaimer**  
**\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!**  
**The data in this folder is strictly confidential: do not share and do not use it without explicit consent from Politecnico di Torino\!**

4 neoplastic subtypes of Renal Cell Carcinoma (RCC):

* clear cell (ccRCC)  
* papillary (pRCC)  
* chromophobe (CHROMO)  
* oncocytoma (ONCO)

For ccRCC and pRCC corresponding .xml files in the respective folder allow obtaining homogeneous regions in terms of depicted class.  
The folder *pre* contains other patients for ccRCC and pRCC classes. The is no difference between *pre*ccRCC and ccRCC as well as between *pre*pRCC and pRCC.  
For ONCO and CHROMO, the corresponding Annotation\_*{}* folder contains homogeneous ROI .svs files. These multiresolution images are cropped from the original WSI and depict (rougly) only one type of tissue. 

The generic file id is the following

HP02.10180.1A2.ccRCC.scn where:

- HP02.10180 is the patient   
- 1A2 is a WSI (a single patient may have more than one corresponding WSI)  
- ccRCC is the diagnosis (remember that in the slide different types of tissue may coexist together)   
- .scn is the WSI file format

The corresponding .xml file *HP02.10180.1A2.ccRCC.xml* identifies a region homogeneous in terms of tissue (in this case an area where the tissue is actually ccRCC class)

Consider the software [ASAP](https://computationalpathologygroup.github.io/ASAP/) or [QuPath](https://github.com/qupath/qupath/releases/) to visualise the data

Ignore the folder README.
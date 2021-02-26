# -*- coding: utf-8 -*-
"""
@author: Steven Court
Methods to apply HU overrides directly to image pixels. Should:
    (i)  Replace all pixels outside external with air HU=-1000
   (ii)  Replace all pixels within a specified structure

"""

import itk
import pydicom
#import gatetools.roi_utils as rt
import roiutils


########################
HU_AIR = -1000
########################


    

def get_external_name( structure_file ):
    """Get contour name of external patient contour"""
    contour = ""
    contains_bolus = False
    ss = pydicom.dcmread( structure_file )    
    for struct in ss.RTROIObservationsSequence:
        if struct.RTROIInterpretedType.lower() == "external":
            contour = struct.ROIObservationLabel
            #print("Found external: {}".format(contour))
        elif struct.RTROIInterpretedType.lower() == "bolus":
            print("\n\nWARNING: Bolus found. It will be overriden with air.\n")
    if contour=="":
        raise Exception("No external structure found. Exiting.")
        exit(1)
    return contour



def set_air_external( img_file, structure_file, output_img_file ):
    """Set all HUs outside of BODY/EXTERNAL contour to air HU=-1000
    
    The img_file must be the .mhd
    """

    img = itk.imread( img_file )
    ds = pydicom.dcmread( structure_file )
    
    contour = get_external_name( structure_file )
  
    
    # MODIFYING GATETOOLS; get_mask() disn't work for HFP setup
    aroi = roiutils.region_of_interest(ds,contour)
    mask = aroi.get_mask(img, corrected=False)
    #itk.imwrite(mask, "mask.mhd")  

    
    pix_mask = itk.array_view_from_image(mask)
    pix_img = itk.array_view_from_image(img) 
    
    if( pix_mask.shape!=pix_img.shape ):
        print( "Inconsistent shapes of mask and image"  )
    
    pix_img_flat = pix_img.flatten()
    for i,val in enumerate( pix_mask.flatten() ):
        if val==0:
            pix_img_flat[i] = HU_AIR
    pix_img = pix_img_flat.reshape( pix_img.shape )
    img_modified = itk.image_view_from_array( pix_img )
    
    img_modified.CopyInformation(img)

    itk.imwrite(img_modified, output_img_file )





def override_hu( img_file, structure_file, output_img, structure, hu ):   #MAYBE JUST PASS THE IMAGE OBJECT AND DICOM OBJECT??
    """Override all HUs inside of specified structure"""

    img = itk.imread( img_file )
    ds = pydicom.dcmread( structure_file )
  
    
    aroi = roiutils.region_of_interest(ds,structure)
    mask = aroi.get_mask(img, corrected=False)    
 
    pix_mask = itk.array_view_from_image(mask)
    pix_img = itk.array_view_from_image(img) 
    if( pix_mask.shape!=pix_img.shape ):
        print( "Inconsistent shapes of mask and image"  )
    
    pix_img_flat = pix_img.flatten()
    for i,val in enumerate( pix_mask.flatten() ):
        if val==1:
            pix_img_flat[i] = hu
    pix_img = pix_img_flat.reshape( pix_img.shape )
    img_modified = itk.image_view_from_array( pix_img )
    
    img_modified.CopyInformation(img)

    itk.imwrite(img_modified, output_img )


# -*- coding: utf-8 -*-
"""
@author: Steven Court

Methods for converting Gate's mhd output to dicom dose file

  - MHD "Offset" = DCM "ImagePositionPatient" (list correct order? x,y,z?)
  - MHD "ElementSpacing" = DCM "PixelSpacing" & GridFrameOffsetVector
  - TransformMatrix is ITK direction
  - MHD=  ?   DCM = ImageOrientationPatient
  - MHD "DimSize" = DCM "pixel_array.shape"
  - axis ordering same? i.e. (X,Y,Z) vs (Z,X,Y)/(Z,Y,X) check this! TODO  
"""

from os import listdir
from os.path import join, isfile, dirname
import random

import itk
import pydicom




def get_dcm_file_path( outputdir, beamref ):
    """ Return path to beam's corresponding dicom dose file
    
    Eclipse dose dcm files in /data directory, one back from outputdir
    """
    parent = dirname(outputdir)
    datadir = join(parent,"data")
    
    filepaths = [join(datadir,f) for f in listdir(datadir) if isfile(join(datadir,f))]
    dcmpaths = [f for f in filepaths if f[-4:]==".dcm" ]

    dcmfile = None
    for dcm in dcmpaths:
        dcmdose = pydicom.dcmread(dcm)
        
        if dcmdose.Modality=="RTDOSE":
            if hasattr( dcmdose.ReferencedRTPlanSequence[0], "ReferencedFractionGroupSequence" ):
                if dcmdose.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0].ReferencedBeamSequence[0].ReferencedBeamNumber == beamref:
                    dcmfile = dcm
            else:
                print(" -- possible plan dose found rather than field?  {}".format(dcm)  ) 
                ##TODO - WHY IS THIS MISSING? BECAUSE IT WAS A PLAN DOSE AND NOT FIELD DOSE?
                #dcmfile = dcm

    if dcmfile is None:
        print(" !! Corresponding dicom dose file not found !!")
    else:
        return dcmfile
        



def rand_digits_str( digits_to_modify ):
    """
    Return a zero-padded random 4 digit string to modify the dicom UIDs
    """
    limit = 10**digits_to_modify - 1
    return str( int(random.random()*limit) ).zfill(digits_to_modify)




def mhd2dcm(mhdFile, dcmFile, output, dosescaling=None):
    """
    Takes mhd dose from Gate and the dicom dose file corresponding field,
    modifes appropriate dicom fields for import into Eclipse.
    
    Optional scaling of dose    
    """
    
    if dosescaling==None:
        #print("No dose scaling specified")
        dosescaling = 1
    
    dcm = pydicom.dcmread(dcmFile)
    mhd=None
    if type(mhdFile)==str:
        mhd = itk.imread(mhdFile)
    else:
        #Assume image
        mhd = mhdFile  ##TODO: TIDY THIS
    
    ###### Alter UID tags  -- TODO: what exactly needs changed?
    digits_to_modify = 4
    digits = rand_digits_str( digits_to_modify )
    
    sopinstanceuid = dcm.SOPInstanceUID 
    dcm.SOPInstanceUID = sopinstanceuid[:-digits_to_modify] + digits
    
    studyinstanceuid = dcm.StudyInstanceUID 
    dcm.StudyInstanceUID = studyinstanceuid[:-digits_to_modify] + digits
    
    seriesinstanceuid = dcm.SeriesInstanceUID 
    dcm.SeriesInstanceUID = seriesinstanceuid[:-digits_to_modify] + digits
    #####################################################
    
    dcm.PixelSpacing = list( mhd.GetSpacing() )[0:2]
    dcm.ImagePositionPatient =  list( mhd.GetOrigin() )
    
    d = mhd.GetDirection() * [1,1,1]
    dcm.ImageOrientationPatient = [ d[0],0,0, 0,d[1],0 ]

    mhdpix = itk.array_from_image(mhd)
    
    dcm.NumberOfFrames = mhdpix.shape[0]
    dcm.Rows = mhdpix.shape[1]            
    dcm.Columns = mhdpix.shape[2]
    
    #Is GridFrameOffsetVector always in "relative interpretations"? 
    # TODO: check this is safe
    # TODO: should this ever negative? - NO!
    dcm.GridFrameOffsetVector = [ x*mhd.GetSpacing()[2] for x in range(mhdpix.shape[0]) ]
         
    dose_abs = mhdpix * dosescaling
    
    scale_to_int = 1E4   # Dicom stores array of integers only
    mhd_scaled = dose_abs * scale_to_int
    mhd_scaled = mhd_scaled.astype(int)       
    dcm.PixelData = mhd_scaled.tobytes()
    
    dcm_scaling = 1.0/scale_to_int
    dcm.DoseGridScaling = dcm_scaling  # Divide dicom's ints by this for dose
    
    #doseunits = dcm.DoseUnits
    #dosescaling = dcm.DoseGridScaling
    #FrameIncrementPointer ?
   
    dcm.save_as( output )


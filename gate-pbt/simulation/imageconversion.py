# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:17:18 2019
@author: SCOURT01

Methods converting CT and mhd images and dose images

"""
import os
import sys

import itk

from gatetools.image_convert import read_dicom



def dcm2mhd_gatetools(ct_files):
    """Testing gatetools; matches below method"""
    itk_img = read_dicom( ct_files )
    itk.imwrite(itk_img, "ct_gatetools.mhd") 



def dcm2mhd( dirName, output ):
    """Convert series of 2D dicom images to mhd + raw files

    Code taken from: https://itk.org/ITKExamples/src/IO/GDCM/ReadDICOMSeriesAndWrite3DImage/Documentation.html

    Need to aovid case of multiple seriesUIDs being found: if dose is different
    from CT then wrong voxel sizes / image dims will be used
    """

    ## Using MET_SHORT will half the file size in comparison to
    ## MET_FLOAT. Allows values -32768 to 32767. 
    ## Is this sufficient for an extended CT range?
    PixelType = itk.ctype('signed short')  
    #PixelType = itk.ctype('float')  
    Dimension = 3
    ImageType = itk.Image[PixelType, Dimension]
    
    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(dirName)
    
    seriesUID = namesGenerator.GetSeriesUIDs()
    
    if len(seriesUID) < 1:
        print('No DICOMs in: ' + dirName)
        sys.exit(1)
    
    print('The directory: ' + dirName)
    print('Contains the following DICOM Series: ')
    for uid in seriesUID:
        print(uid)
    
    seriesFound = False
    for uid in seriesUID:
        seriesIdentifier = uid
        print('Reading: ' + seriesIdentifier)
        fileNames = namesGenerator.GetFileNames(seriesIdentifier)
    
        reader = itk.ImageSeriesReader[ImageType].New()
        dicomIO = itk.GDCMImageIO.New()
        reader.SetImageIO(dicomIO)
        reader.SetFileNames(fileNames)
    
        writer = itk.ImageFileWriter[ImageType].New()
    
        outFileName = os.path.join(output)
    
        writer.SetFileName(outFileName)
        writer.SetInput(reader.GetOutput())
        print('Writing: ' + outFileName)
        writer.Update()
    
        if seriesFound:
            break


       


#def dose_dcm2mhd_norm(ds):
#    """
#    Method to convert a dcm dose file into mhd+raw image 
#    Dose is normalized to max
#    """            
#    # Must cast to numpy.float32 for ITK to write mhd with ElementType MET_FLOAT
#    px_data = ds.pixel_array.astype(np.float32)   
#    
#    print( "\n***Generating mhd from dose dicom ***")
#    print( "  PixelData shape = {}".format(px_data.shape) )
#
#    # DoseGridScaling * pixel_value gives dose in DoseUnits
#    # Gives total plan dose for field, not fraction dose.
#    scale = ds.DoseGridScaling   
#    units = ds.DoseUnits 
#    print("  DoseUnits: {}".format(units))  
#    print("  DoseGridScale: {}".format(scale))
#    print("  DoseImageDims: {}".format(px_data.shape))
#
#    print("  Scaling dose to {}".format(units) )
#    px_data = px_data * scale     
#    print("  Max scaled dose = {}Gy".format( np.max(px_data) ) )  
#    
#    # TODO: Can I normalize or should we be comparing absolute doses?
#    # Gate DoseActor gives us option to normalize to max so we could compare these?
#    mx = np.max( px_data )
#    px_data = px_data / mx
#        
#    imgpospat = ds.ImagePositionPatient
#    xy_spacing = ds.PixelSpacing
#    ##orientation = ds.ImageOrientationPatient?
#    z_spacing = ds.GridFrameOffsetVector[1]-ds.GridFrameOffsetVector[0]  ##TODO: assumed these are equal but they might not be with our scanner
#    pixelspacing = [xy_spacing[0],xy_spacing[1]] + [z_spacing]
#    
#    # 3D ITK image 
#    doseimg = itk.image_from_array( px_data )
#    doseimg.SetOrigin( imgpospat )
#    doseimg.SetSpacing( pixelspacing )
#    ##doseimg.SetDirection( orientation )
#    
#    itk.imwrite(doseimg, "EclipseDose.mhd")


#
#def rand_digits_str( digits_to_modify ):
#    """
#    Return a zero-padded random 4 digit string to modify the dicom UIDs
#    """
#    limit = 10**digits_to_modify - 1
#    return str( int(random.random()*limit) ).zfill(digits_to_modify)
#    


#
#def dose_mhd2dcm(mhdFile, dcmFile, prescription=None):
#    """
#    Takes mhd dose from Gate and a dicom dose file (corresponding field)
#    and modifes appropriate dicom fields for import into Eclipse.
#    """
#    # Scale the plan's dose to prescription so that it's in the right ball-park
#    if prescription==None:
#        prescription = 50.0
#    
#    dcm = pydicom.dcmread(dcmFile)
#    mhd = itk.imread(mhdFile)
#    
#    ###### Alter UID tags  -- WHAT DO I NEED TO CHANGE?
#    digits_to_modify = 4
#    digits = rand_digits_str( digits_to_modify )
#    
#    sopinstanceuid = dcm.SOPInstanceUID 
#    dcm.SOPInstanceUID = sopinstanceuid[:-digits_to_modify] + digits
#    
#    studyinstanceuid = dcm.StudyInstanceUID 
#    dcm.StudyInstanceUID = studyinstanceuid[:-digits_to_modify] + digits
#    
#    seriesinstanceuid = dcm.SeriesInstanceUID 
#    dcm.SeriesInstanceUID = seriesinstanceuid[:-digits_to_modify] + digits
#    #####################################################
#    
#       
#    ###### Alter the necessary image properties
#    '''
#    - MHD "Offset" = DCM "ImagePositionPatient" (list correct order? x,y,z?)
#    - MHD "ElementSpacing" = DCM "PixelSpacing" & GridFrameOffsetVector
#    - TransformMatrix always "1 0 0 0 1 0 0 0 1"?
#    - MHD=  ??????   DM = ImageOrientationPatient
#    - MHD "DimSize" = DCM "pixel_array.shape"
#    - axis ordering same? i.e. (X,Y,Z) vs (Z,X,Y)/(Z,Y,X) check this!
#    '''
#    
#    dcm.PixelSpacing = list( mhd.GetSpacing() )[0:2]
#    dcm.ImagePositionPatient =  list( mhd.GetOrigin() )
#
#    # itk.array_from_image(mhd) = DCM ".pixel_array"
#    mhdpix = itk.array_from_image(mhd)
#    
#    dcm.NumberOfFrames = mhdpix.shape[0]
#    dcm.Rows = mhdpix.shape[1]            ## ARE THESE CORRECT WAY ROUND?
#    dcm.Columns = mhdpix.shape[2]
#    
#    #Is GridFrameOffsetVector always in "relative interpretations"?
#    dcm.GridFrameOffsetVector = [ x*mhd.GetSpacing()[2] for x in range(mhdpix.shape[0]) ]
#    
#    # No - we want to scale all by same amount so that in Eclipse we can scale
#    #  the dose and LET-weighted doses by the same factor.
#    ###MAX_UINT32 = 2147483647
#    ###scale_to_int = (MAX_UINT32/mhdpix.max())*0.8 
#    #
#    ###dcmvals = dcm.pixel_array  
#    ###PRESCRIPTION_GY = prescription
#    ###scale_to_presc = PRESCRIPTION_GY*1.0/dcmvals.max()
#     
#    scale_to_int = 1E8
#    #scale_to_int = 1.0E4 ## NEED THIS IS YOU'RE DOING STRAIGHT LET (keV/um)
#    mhd_scaled = mhdpix*scale_to_int
#    mhd_scaled = mhd_scaled.astype(int)       
#    dcm.PixelData = mhd_scaled.tobytes()
#    
#    scale_to_presc = 1.0E-6
#    #scale_to_presc = 1.0E-4 ## NEED THIS IS YOU'RE DOING STRAIGHT LET (keV/um)    
#    dcm.DoseGridScaling = scale_to_presc
#    
#    #doseunits = dcm.DoseUnits
#    #dosescaling = dcm.DoseGridScaling
#    #FrameIncrementPointer ???????
#   
#    dcm.save_as( "zmhd2dcm_dose.dcm")
#
#
#
#
#def dose_mhd2dcm_scale(mhdFile, dcmFile, dosescaling=None):
#    """
#    Takes mhd dose from Gate and a dicom dose file (corresponding field)
#    and modifes appropriate dicom fields for import into Eclipse.
#    
#    Scales by Nreq/Nsim factor for absilute dose
#    
#    """
#    # Scale the plan's dose to prescription so that it's in the right ball-park
#    if dosescaling==None:
#        print(" NO DOSE SCALING SPECIFIED")
#        dosescaling = 1
#    
#    dcm = pydicom.dcmread(dcmFile)
#    mhd = itk.imread(mhdFile)
#    
#    ###### Alter UID tags  -- WHAT DO I NEED TO CHANGE?
#    digits_to_modify = 4
#    digits = rand_digits_str( digits_to_modify )
#    
#    sopinstanceuid = dcm.SOPInstanceUID 
#    dcm.SOPInstanceUID = sopinstanceuid[:-digits_to_modify] + digits
#    
#    studyinstanceuid = dcm.StudyInstanceUID 
#    dcm.StudyInstanceUID = studyinstanceuid[:-digits_to_modify] + digits
#    
#    seriesinstanceuid = dcm.SeriesInstanceUID 
#    dcm.SeriesInstanceUID = seriesinstanceuid[:-digits_to_modify] + digits
#    #####################################################
#    
#       
#    ###### Alter the necessary image properties
#    '''
#    - MHD "Offset" = DCM "ImagePositionPatient" (list correct order? x,y,z?)
#    - MHD "ElementSpacing" = DCM "PixelSpacing" & GridFrameOffsetVector
#    - TransformMatrix always "1 0 0 0 1 0 0 0 1"?
#    - MHD=  ??????   DM = ImageOrientationPatient
#    - MHD "DimSize" = DCM "pixel_array.shape"
#    - axis ordering same? i.e. (X,Y,Z) vs (Z,X,Y)/(Z,Y,X) check this!
#    '''
#    
#    dcm.PixelSpacing = list( mhd.GetSpacing() )[0:2]
#    dcm.ImagePositionPatient =  list( mhd.GetOrigin() )
#
#    # itk.array_from_image(mhd) = DCM ".pixel_array"
#    mhdpix = itk.array_from_image(mhd)
#    
#    dcm.NumberOfFrames = mhdpix.shape[0]
#    dcm.Rows = mhdpix.shape[1]            ## ARE THESE CORRECT WAY ROUND?
#    dcm.Columns = mhdpix.shape[2]
#    
#    #Is GridFrameOffsetVector always in "relative interpretations"?
#    dcm.GridFrameOffsetVector = [ x*mhd.GetSpacing()[2] for x in range(mhdpix.shape[0]) ]
#    
#    # No - we want to scale all by same amount so that in Eclipse we can scale
#    #  the dose and LET-weighted doses by the same factor.
#    ###MAX_UINT32 = 2,147,483,647
#    ###scale_to_int = (MAX_UINT32/mhdpix.max())*0.8 
#    #
#    ###dcmvals = dcm.pixel_array  
#    ###PRESCRIPTION_GY = prescription
#    ###scale_to_presc = PRESCRIPTION_GY*1.0/dcmvals.max()
#     
#    dose_abs = mhdpix * dosescaling
#    
#    scale_to_int = 1E4
#    #scale_to_int = 1.0E4 ## NEED THIS IS YOU'RE DOING STRAIGHT LET (keV/um)
#    mhd_scaled = dose_abs * scale_to_int
#    mhd_scaled = mhd_scaled.astype(int)       
#    dcm.PixelData = mhd_scaled.tobytes()
#    
#    dcm_scaling = 1.0/scale_to_int
#    dcm.DoseGridScaling = dcm_scaling
#    
#    #doseunits = dcm.DoseUnits
#    #dosescaling = dcm.DoseGridScaling
#    #FrameIncrementPointer ???????
#   
#    dcm.save_as( "zmhd2dcm_dose.dcm")







### https://discourse.itk.org/t/dividereal-always-returns-double-image/282



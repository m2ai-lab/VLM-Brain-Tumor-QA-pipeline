# Import all necessary libraries
import pandas as pd
import pydicom
import openpyxl
import nibabel as nifti
import numpy as np
import os

dicom_data = pd.DataFrame()

#get dicom folder path
dicom_path = "path/to/dicoms"
dicom_files = os.listdir(dicom_path)
nifti_path = "path/to/niftis"

#get all of the .dicom files
dicom_files = [files for files in dicom_files 
               if files.endswith('.dcm')]

#Empty dataset of files
dicom_data = pd.DataFrame({"files": []})

for file in dicom_files:
    # Read the DICOM file
    data = pydicom.dcmread(dicom_path + "/" + file)

    #Setting voxels up for 
    voxel_spacing = [1.0, 1.0, 1.0]
    affine = np.diag([voxel_spacing[2], voxel_spacing[1], voxel_spacing[0], 1])

    #Save the nifti files in the specific folders
    nifti_image = nifti.Nifti1Image(data.pixel_array, affine)
    nifti.save(nifti_image, nifti_path+ "/" + file[:-4]+'.nii')

    #Add data to dictionary 
    dicom_data = pd.concat([dicom_data, pd.DataFrame([{"files":
             [nifti_path+ "/" +file[:-4]+'.nii']}])],
             ignore_index=True)

dicom_data.to_excel('test_data.xlsx', engine='openpyxl')

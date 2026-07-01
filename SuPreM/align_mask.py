import numpy as np
import nibabel as nib
from pathlib import Path
import argparse

ref_img = nib.load("/home/s2347484/Seg/testing_set/UKCHLL007/image.nii.gz")
seg_img = nib.load("/home/s2347484/Seg/testing_set/UKCHLL007/annotation_2.nii.gz")

correct_affine = ref_img.affine
correct_zooms = ref_img.header.get_zooms()

seg_data = seg_img.get_fdata()
corrected_seg = nib.Nifti1Image(seg_data, affine=correct_affine)
corrected_seg.header.set_zooms(correct_zooms)

nib.save(corrected_seg, "/home/s2347484/Seg/testing_set/UKCHLL007/annotation_2_corrected.nii.gz")


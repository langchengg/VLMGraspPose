from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .aligned_crops import CROP_CHANNELS
from .coordinate_mapping import axial_difference_deg


DERIVED_HEAD_FEATURE_NAMES = (
    "g0_nearest_higher_peak_distance_fraction",
    "g0_predicted_mask_centroid_distance_fraction",
    "g0_predicted_mask_principal_axis_difference_fraction",
    "g3_rho_roi_min",
    "g3_candidate_contact_angle_difference_fraction",
    "g4_contact_band_width_probability_mean",
)


def derived_head_features(
    crop: np.ndarray,
    candidate: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    *,
    image_shape: tuple[int, int],
) -> np.ndarray:
    array=np.asarray(crop,dtype=np.float64)
    mask=array[CROP_CHANNELS.index("mask_probability")]
    sin2=array[CROP_CHANNELS.index("sin_2theta")]
    cos2=array[CROP_CHANNELS.index("cos_2theta")]
    width=array[CROP_CHANNELS.index("width_probability")]
    height,width_px=map(float,image_shape); diagonal=max(math.hypot(height,width_px),1.0)
    rank=int(candidate["q_rank"])
    higher=[value for value in candidates if int(value["q_rank"])<rank]
    nearest=0.0 if not higher else min(math.hypot(float(candidate["cx"])-float(value["cx"]),float(candidate["cy"])-float(value["cy"])) for value in higher)/diagonal
    size=mask.shape[-1]; axis=np.linspace(-1.0,1.0,size); vv,uu=np.meshgrid(axis,axis,indexing="ij")
    mass=float(np.clip(mask,0.0,None).sum())
    if mass>1e-8:
        centroid_u=float((mask*uu).sum()/mass); centroid_v=float((mask*vv).sum()/mass)
        centroid_distance=math.hypot(centroid_u,centroid_v)/math.sqrt(2.0)
        du=uu-centroid_u; dv=vv-centroid_v
        covariance=np.asarray([[(mask*du*du).sum(),(mask*du*dv).sum()],[(mask*du*dv).sum(),(mask*dv*dv).sum()]],dtype=np.float64)/mass
        eigenvalues,eigenvectors=np.linalg.eigh(covariance); vector=eigenvectors[:,int(np.argmax(eigenvalues))]
        principal_angle=math.degrees(math.atan2(float(vector[1]),float(vector[0])))
        principal_difference=axial_difference_deg(principal_angle,0.0)/90.0
    else:
        centroid_distance=1.0; principal_difference=1.0
    rho=np.hypot(sin2,cos2)
    contact=(np.abs(uu)>=.35)&(np.abs(uu)<=.75)&(np.abs(vv)<=.24)
    weights=np.clip(rho[contact],1e-8,None)
    contact_s=float(np.sum(weights*np.sin(np.arctan2(sin2[contact],cos2[contact]))))
    contact_c=float(np.sum(weights*np.cos(np.arctan2(sin2[contact],cos2[contact]))))
    contact_angle=0.5*math.degrees(math.atan2(contact_s,contact_c))
    candidate_contact=axial_difference_deg(float(candidate["angle_deg"]),contact_angle)/90.0
    result=np.asarray((nearest,centroid_distance,principal_difference,float(np.min(rho)),candidate_contact,float(np.mean(width[contact]))),dtype=np.float32)
    if not np.isfinite(result).all(): raise FloatingPointError("non-finite derived full-chain head feature")
    return result


__all__=("DERIVED_HEAD_FEATURE_NAMES","derived_head_features")

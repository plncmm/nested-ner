import minio
import pathlib
import logging
import argparse


def samples_loader_from_minio(server,access_key,secret_key, region, n_annotations, verbose=False, return_filename=True):
    """
    Code adapted from https://github.com/fvillena/wl-corpus/blob/master/describe_samples.py. It returns an array 
    with referrals content and itsannotation. The files are obtained from a remote object server of the research group.
    """
    minio_client = minio.Minio(
        server,
        access_key=access_key,
        secret_key=secret_key,
        region=region,
        secure=True,
    )
    objects = minio_client.list_objects("brat-data",prefix='wl_ground_truth/')
    referrals = []
    annotations = []
    cnt = 0
    for o in objects:
        if o.object_name.endswith(".txt") and cnt<n_annotations:
            cnt+=1
            ann_filepath = f"{o.object_name[:-4]}.ann"
            txt_object = minio_client.get_object("brat-data", o.object_name)
            ann_object = minio_client.get_object("brat-data", ann_filepath)
            txt_name = pathlib.Path(o.object_name).name
            ann_name = pathlib.Path(ann_filepath).name
            if verbose: logger.info(f"Processing referral: {txt_name}")
            if return_filename:
                referrals.append((txt_name,txt_object.read().decode("utf-8")))
                annotations.append((ann_name,ann_object.read().decode("utf-8")))
            else:
                referrals.append(txt_object.read().decode("utf-8"))
                annotations.append(ann_object.read().decode("utf-8"))
    assert(len(referrals)==len(annotations))
    print(f"{len(referrals)} samples were downloaded")
    return referrals, annotations



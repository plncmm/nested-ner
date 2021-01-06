import minio
import pathlib
import logging
import argparse 

def windows_linux_ambiguity(referral, annotation, referral_name):
    for line in annotation.splitlines():
        entity = line.split()
        if entity[0].startswith('T') and not ';' in entity[3] and ' '.join(referral[int(entity[2]): int(entity[3])].split())!=' '.join(entity[4:]):
            return True
    return False

def samples_loader_from_minio(server,access_key,secret_key, n_annotations):
    """
    Code adapted from https://github.com/fvillena/wl-corpus/blob/master/describe_samples.py. It returns an array 
    with referrals content and its annotation. The files are obtained from a remote object server of the research group.
    """
    referrals = []
    annotations = []
    idx = 0

    # Connecting to the object server and get annotations objects.
    minio_client = minio.Minio(
        server,
        access_key=access_key,
        secret_key=secret_key,
        region='cl',
        secure=True,
    )
    objects = minio_client.list_objects("brat-data", prefix='wl_ground_truth/')

    for o in objects:
        if o.object_name.endswith(".txt") and idx < n_annotations:
            ann_filepath = f"{o.object_name[:-4]}.ann"
            txt_object = minio_client.get_object("brat-data", o.object_name)
            ann_object = minio_client.get_object("brat-data", ann_filepath)
            txt_name = pathlib.Path(o.object_name).name
            ann_name = pathlib.Path(ann_filepath).name
            txt_content = txt_object.read().decode("utf-8")
            ann_content = ann_object.read().decode("utf-8")
            if windows_linux_ambiguity(txt_content, ann_content, txt_name):
                continue
            referrals.append((txt_name,txt_content))
            annotations.append((ann_name,ann_content))
            idx+=1
    print(f"{len(referrals)} samples were downloaded")
    return referrals, annotations



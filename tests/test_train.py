from multipose2 import io, models, train
from subprocess import check_output, STDOUT
import os, shutil
import torch
from pathlib import Path
import numpy as np


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def test_train_channel_guard_rebuilds_adapter():
    model = models.CellposeModel(gpu=False, nchan=3)
    train._ensure_net_input_channels(model.net, 5)
    assert model.net.in_channels == 5


def test_synthesize_multimodal_training_dir(tmp_path):
    he_dir = tmp_path / "H&EStain" / "Training"
    tx_dir = tmp_path / "UnremovedTranscripts" / "Training"
    out_dir = tmp_path / "Synthesized" / "Training"
    he_dir.mkdir(parents=True)
    tx_dir.mkdir(parents=True)

    io.imsave(str(he_dir / "sample_001.tif"),
              np.zeros((16, 16, 3), dtype=np.float32))
    io.imsave(str(tx_dir / "sample_001.tif"),
              np.ones((16, 16, 2), dtype=np.float32))
    io.imsave(str(he_dir / "sample_001_masks.tif"),
              np.zeros((16, 16), dtype=np.uint16))

    train_dir = io.synthesize_multimodal_training_dir(
        modality_dirs={"he": he_dir, "transcripts": tx_dir},
        output_dir=out_dir,
        label_dir=he_dir,
        mask_filter="_masks.tif",
        modality_channel_axes={"he": -1, "transcripts": -1},
    )

    images, labels, image_names, *_ = io.load_train_test_data(
        str(train_dir), mask_filter="_masks.tif"
    )
    assert len(images) == 1
    assert images[0].shape == (16, 16, 5)
    assert labels[0].shape == (16, 16)
    assert Path(image_names[0]).parent == out_dir


def test_class_train(data_dir):
    train_dir = str(data_dir.joinpath('2D').joinpath('train'))
    model_dir = str(data_dir.joinpath('2D').joinpath('train').joinpath('models'))
    shutil.rmtree(model_dir, ignore_errors=True)
    output = io.load_train_test_data(train_dir, mask_filter='_cyto_masks')
    images, labels, image_names, test_images, test_labels, image_names_test = output
    use_gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=use_gpu)
    cpmodel_path = train.train_seg(model.net, images, labels, train_files=image_names,
                                   test_data=test_images, test_labels=test_labels,
                                   test_files=image_names_test,
                                   save_path=train_dir, n_epochs=3)[0]
    io.add_model(cpmodel_path)
    io.remove_model(cpmodel_path, delete=True)
    print('>>>> model trained and saved to %s' % cpmodel_path)


def test_cli_train(data_dir):
    # import sys
    # path_root = Path(__file__).parents[1]
    # sys.path.append(str(path_root))
    # print(Path(__file__).parents[0],Path(__file__).parents[1],Path(__file__).parents[2])
    train_dir = str(data_dir.joinpath('2D').joinpath('train'))
    model_dir = str(data_dir.joinpath('2D').joinpath('train').joinpath('models'))
    shutil.rmtree(model_dir, ignore_errors=True)
    use_gpu = torch.cuda.is_available()
    gpu_str = "--use_gpu" if use_gpu else ""
    cmd = 'python -m multipose2 %s --train --n_epochs 3 --dir %s --mask_filter _cyto_masks --pretrained_model None' % (gpu_str, train_dir)
    try:
        cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
    except Exception as e:
        print(e)
        raise ValueError(e)


def test_cli_make_train(data_dir):
    script_name = Path().resolve() / 'multipose2/gui/make_train.py'
    image_path = data_dir / '3D/gray_3D.tif'

    cmd = f'python {script_name} --image_path {image_path}'
    res = check_output(cmd, stderr=STDOUT, shell=True)

    # there should be 30 slices: 
    files = [f for f in (data_dir / '3D/train/').iterdir() if 'gray_3D' in f.name]
    assert 30 == len(files)

    shutil.rmtree((data_dir / '3D/train'))

#!/bin/bash

# Parameters
while getopts g:v:s:e: option; do
    case "${option}" in
        g) gpu=${OPTARG};;
        v) video=${OPTARG};;
        s) start=${OPTARG};;
        e) end=${OPTARG};;
    esac
done

# Configration
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES="${gpu:-0}"
VIDEO_ID=${video:-"May"}  # path: ./data/raw/videos/${VIDEO_ID}.mp4 (avoid using full numbers as file name)
AUDIO_ID="demo"  # path: ./data/raw/val_wavs/${AUDIO_ID}.wav

lpips_start_iters=140_000
max_updates=150_000

start_step="${start:-1}"
stop_step="${end:-3}"

echo "CUDA: ${CUDA_VISIBLE_DEVICES}"
echo "Video: ${VIDEO_ID}"
echo "Audio: ${AUDIO_ID}"
echo "start_step: ${start_step}"
echo "stop_step: ${stop_step}"

# Generated dir:
# ./data/raw/videos/${VIDEO_ID}_orig.mp4
# ./data/binary/videos/${VIDEO_ID}
# ./data/processed/videos/${VIDEO_ID}
# ./egs/datasets/${VIDEO_ID}
# ./checkpoints/motion2video_nerf/${VIDEO_ID}_head
# ./checkpoints/motion2video_nerf/${VIDEO_ID}_torso

# Cache files:
# (HuBERT_Wav2Vec2)
# /home/lcs/.cache/huggingface/hub/models--facebook--hubert-large-ls960-ft/snapshots/ece5fabbf034c1073acae96d5401b25be96709d8
# (VGG)
# /home/lcs/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth
# /home/lcs/.cache/torch/hub/checkpoints/vgg_face_dag.pth


if [[ ${start_step} -le 1 ]] && [[ ${stop_step} -ge 1 ]]; then
    echo "======================================== Data Pre-process ========================================"

    echo "---------------------------------------- 1.1. Crop videos to 512 * 512 and 25 FPS ----------------------------------------"
    crop_file="./data/raw/videos/${VIDEO_ID}_orig.mp4"
    if [ -e "$crop_file" ]; then
        echo "Video file is already cropped!"
    else
        ffmpeg -loglevel panic -y -i "./data/raw/videos/${VIDEO_ID}.mp4" -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 "./data/raw/videos/${VIDEO_ID}_512.mp4"
        mv "./data/raw/videos/${VIDEO_ID}.mp4" "./data/raw/videos/${VIDEO_ID}_orig.mp4"
        mv "./data/raw/videos/${VIDEO_ID}_512.mp4" "./data/raw/videos/${VIDEO_ID}.mp4"
    fi

    echo "---------------------------------------- 1.2. Prepare config files ----------------------------------------"
    mkdir -p "egs/datasets/${VIDEO_ID}"
    cp ./egs/datasets/May/* "./egs/datasets/${VIDEO_ID}/"
    # change video name
    sed -i "s/May/${VIDEO_ID}/g" "./egs/datasets/${VIDEO_ID}/lm3d_radnerf.yaml"
    sed -i "s/May/${VIDEO_ID}/g" "./egs/datasets/${VIDEO_ID}/lm3d_radnerf_torso.yaml"
    # change number of training iters
    sed -i "s/20_0000/${lpips_start_iters}/g" "./egs/datasets/${VIDEO_ID}/lm3d_radnerf_sr.yaml"
    sed -i "s/25_0000/${max_updates}/g" "./egs/datasets/${VIDEO_ID}/lm3d_radnerf_sr.yaml"
    sed -i "s/20_0000/${lpips_start_iters}/g" "./egs/datasets/${VIDEO_ID}/lm3d_radnerf_torso_sr.yaml"
    sed -i "s/25_0000/${max_updates}/g" "./egs/datasets/${VIDEO_ID}/lm3d_radnerf_torso_sr.yaml"
    sed -i "s/num_updates/max_updates/g" "./egs/datasets/${VIDEO_ID}/lm3d_radnerf_torso_sr.yaml"

    echo "---------------------------------------- 1.3. Extract audio features, such as mel, f0, hubert and esperanto ----------------------------------------"
    mkdir -p "./data/processed/videos/${VIDEO_ID}"
    ffmpeg -loglevel panic -y -i "./data/raw/videos/${VIDEO_ID}.mp4" -f wav -ar 16000 "./data/processed/videos/${VIDEO_ID}/aud.wav"
    python data_gen/utils/process_audio/extract_hubert.py --video_id="${VIDEO_ID}"
    python data_gen/utils/process_audio/extract_mel_f0.py --video_id="${VIDEO_ID}"

    echo "---------------------------------------- 1.4. Extract images ----------------------------------------"
    mkdir -p "./data/processed/videos/${VIDEO_ID}/gt_imgs"
    ffmpeg -loglevel panic -y -i "./data/raw/videos/${VIDEO_ID}.mp4" -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 -start_number 0 "./data/processed/videos/${VIDEO_ID}/gt_imgs/%08d.jpg"
    python data_gen/utils/process_video/extract_segment_imgs.py --ds_name=nerf --vid_dir="./data/raw/videos/${VIDEO_ID}.mp4" --force_single_process  # extract image, segmap, and background

    echo "---------------------------------------- 1.5. Extract lm2d_mediapipe ----------------------------------------"
    python data_gen/utils/process_video/extract_lm2d.py --num_workers=4 --ds_name=nerf --vid_dir="./data/raw/videos/${VIDEO_ID}.mp4"

    echo "---------------------------------------- 1.6. Fit 3DMM ----------------------------------------"
    python data_gen/utils/process_video/fit_3dmm_landmark.py --ds_name=nerf --vid_dir="./data/raw/videos/${VIDEO_ID}.mp4" --reset  --debug --id_mode=global

    echo "---------------------------------------- 1.7. Binarize ----------------------------------------"
    python data_gen/runs/binarizer_nerf.py --video_id="${VIDEO_ID}"
fi


if [[ ${start_step} -le 2 ]] && [[ ${stop_step} -ge 2 ]]; then
    echo "======================================== Training ========================================"

    echo "---------------------------------------- 2.1. Train the Head NeRF ----------------------------------------"
    python tasks/run.py \
    --config="./egs/datasets/${VIDEO_ID}/lm3d_radnerf_sr.yaml" \
    --exp_name="./motion2video_nerf/${VIDEO_ID}_head" \
    --reset

    echo "---------------------------------------- 2.2. Train the Torso NeRF ----------------------------------------"
    python tasks/run.py \
    --config="./egs/datasets/${VIDEO_ID}/lm3d_radnerf_torso_sr.yaml" \
    --exp_name="./motion2video_nerf/${VIDEO_ID}_torso" \
    --hparams="head_model_dir=./checkpoints/motion2video_nerf/${VIDEO_ID}_head" \
    --reset
fi


if [[ ${start_step} -le 3 ]] && [[ ${stop_step} -ge 3 ]]; then
    echo "======================================== Inference ========================================"
    python inference/genefacepp_infer.py \
    --a2m_ckpt="checkpoints/audio2motion_vae" \
    --torso_ckpt="checkpoints/motion2video_nerf/${VIDEO_ID}_torso" \
    --drv_aud="data/raw/val_wavs/${AUDIO_ID}.wav" \
    --out_name="${VIDEO_ID}_${AUDIO_ID}_output.mp4"
fi

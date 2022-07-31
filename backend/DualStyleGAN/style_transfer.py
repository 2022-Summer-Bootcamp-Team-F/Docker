import os
import numpy as np
import torch
from util import save_image, load_image
import argparse
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision

from model.dualstylegan import DualStyleGAN
from model.sampler.icp import ICPTrainer
from model.encoder.psp import pSp

CODE_DIR = 'DualStyleGAN'
device = 'cpu'

# os.getcwd() : 현재 디렉터리 절대경로를 "str"로 반환
MODEL_DIR = os.path.join(os.path.dirname(os.getcwd()), CODE_DIR, 'checkpoint')  # DualStyleGAN/checkpoint
DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), CODE_DIR, 'data')  # DualStyleGAN/data

# Step 1: Select Style Type
style_types = ['cartoon', 'caricature', 'anime', 'arcane', 'comic', 'pixar', 'slamdunk']
style_type = style_types[0]

MODEL_PATHS = {
    "encoder": {"id": "1NgI4mPkboYvYw3MWcdUaQhkr0OWgs9ej", "name": "encoder.pt"},
    "cartoon-G": {"id": "1exS9cSFkg8J4keKPmq2zYQYfJYC5FkwL", "name": "generator.pt"},
    "cartoon-N": {"id": "1JSCdO0hx8Z5mi5Q5hI9HMFhLQKykFX5N", "name": "sampler.pt"},
    "cartoon-S": {"id": "1ce9v69JyW_Dtf7NhbOkfpH77bS_RK0vB", "name": "refined_exstyle_code.npy"},
    "caricature-G": {"id": "1BXfTiMlvow7LR7w8w0cNfqIl-q2z0Hgc", "name": "generator.pt"},
    "caricature-N": {"id": "1eJSoaGD7X0VbHS47YLehZayhWDSZ4L2Q", "name": "sampler.pt"},
    "caricature-S": {"id": "1-p1FMRzP_msqkjndRK_0JasTdwQKDsov", "name": "refined_exstyle_code.npy"},
    "anime-G": {"id": "1BToWH-9kEZIx2r5yFkbjoMw0642usI6y", "name": "generator.pt"},
    "anime-N": {"id": "19rLqx_s_SUdiROGnF_C6_uOiINiNZ7g2", "name": "sampler.pt"},
    "anime-S": {"id": "17-f7KtrgaQcnZysAftPogeBwz5nOWYuM", "name": "refined_exstyle_code.npy"},
    "arcane-G": {"id": "15l2O7NOUAKXikZ96XpD-4khtbRtEAg-Q", "name": "generator.pt"},
    "arcane-N": {"id": "1fa7p9ZtzV8wcasPqCYWMVFpb4BatwQHg", "name": "sampler.pt"},
    "arcane-S": {"id": "1z3Nfbir5rN4CrzatfcgQ8u-x4V44QCn1", "name": "exstyle_code.npy"},
    "comic-G": {"id": "1_t8lf9lTJLnLXrzhm7kPTSuNDdiZnyqE", "name": "generator.pt"},
    "comic-N": {"id": "1RXrJPodIn7lCzdb5BFc03kKqHEazaJ-S", "name": "sampler.pt"},
    "comic-S": {"id": "1ZfQ5quFqijvK3hO6f-YDYJMqd-UuQtU-", "name": "exstyle_code.npy"},
    "pixar-G": {"id": "1TgH7WojxiJXQfnCroSRYc7BgxvYH9i81", "name": "generator.pt"},
    "pixar-N": {"id": "18e5AoQ8js4iuck7VgI3hM_caCX5lXlH_", "name": "sampler.pt"},
    "pixar-S": {"id": "1I9mRTX2QnadSDDJIYM_ntyLrXjZoN7L-", "name": "exstyle_code.npy"},
    "slamdunk-G": {"id": "1MGGxSCtyf9399squ3l8bl0hXkf5YWYNz", "name": "generator.pt"},
    "slamdunk-N": {"id": "1-_L7YVb48sLr_kPpOcn4dUq7Cv08WQuG", "name": "sampler.pt"},
    "slamdunk-S": {"id": "1Dgh11ZeXS2XIV2eJZAExWMjogxi_m_C8", "name": "exstyle_code.npy"},
}

print('*' * 100)

# Step 2: Load Pretrained Model
# We assume that you have downloaded all relevant models and placed them in the directory defined by the above dictionary.

# torchvision.transforms : 다양한 이미지 변환 기능들을 제공
# torchvision.transforms.Compose : transform 들을 Compose로 구성할 수 있습니다.
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

# load DualStyleGAN
generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
generator.eval()
# ckpt : checkpoint
ckpt = torch.load(os.path.join(MODEL_DIR, style_type, 'generator.pt'),
                  map_location=lambda storage, loc: storage)  # DualStyleGAN/checkpoint/style_type/generator.pt
generator.load_state_dict(ckpt["g_ema"])
generator = generator.to(device)

# load encoder
model_path = os.path.join(MODEL_DIR, 'encoder.pt')  # DualStyleGAN/checkpoint/encoder.pt
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = model_path  # DualStyleGAN/checkpoint/encoder.pt
opts = Namespace(**opts)
opts.device = device
encoder = pSp(opts)
encoder.eval()
encoder = encoder.to(device)

# load extrinsic style code
exstyles = np.load(os.path.join(MODEL_DIR, style_type, MODEL_PATHS[style_type + '-S']["name"]),
                   allow_pickle='TRUE').item()  # DualStyleGAN/checkpoint/style_type

# load sampler network
icptc = ICPTrainer(np.empty([0, 512 * 11]), 128)
icpts = ICPTrainer(np.empty([0, 512 * 7]), 128)
ckpt = torch.load(os.path.join(MODEL_DIR, style_type, 'sampler.pt'), map_location=lambda storage, loc: storage)
icptc.icp.netT.load_state_dict(ckpt['color'])
icpts.icp.netT.load_state_dict(ckpt['structure'])
icptc.icp.netT = icptc.icp.netT.to(device)
icpts.icp.netT = icpts.icp.netT.to(device)

print('Model successfully loaded!')  # pretrained 모델 로드 완료

# Step 3: Visualize Input

image_path = './data/content/unsplash-rDEOVtE7vOs.jpg'  # input 이미지 경로, 여기에 s3 이미지 경로를 직접 넣어도 될거 같음
original_image = load_image(image_path)  # input 이미지

# Step 4: Align Image, 이미지 정렬
# Note: Our style transfer assumes the input has been pre-aligned. If the original image is not pre-aligned, please run the following alignment scripts.

if_align_face = True


def run_alignment(image_path):  # image_path : Input 이미지 경로
    import dlib
    from model.encoder.align_all_parallel import align_face
    modelname = os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')  # DualStyleGAN/checkpoint  99.7MB 괜
    # if not os.path.exists(modelname):  # 만약 이 파일도 용량이 크면 S3로 빼야할거 같습니다. 일단 넣는 방향으로 진행, 위에 파일이 없을 경우 실행되는 코드 미리 배치해놓을거기 때문에 코드 변형할 필요 있음
    #     import wget, bz2
    #     wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname + '.bz2')
    #     zipfile = bz2.BZ2File(modelname + '.bz2')
    #     data = zipfile.read()  # 압축 파일 해제
    #     open(modelname, 'wb').write(data)
    predictor = dlib.shape_predictor(modelname)
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    return aligned_image  # 보정된 이미지 반환


# 위에서 if_align_face = True이므로 무조건 실행
if if_align_face:
    I = transform(run_alignment(image_path)).unsqueeze(dim=0).to(device)
else:
    I = F.adaptive_avg_pool2d(load_image(image_path).to(device), 256)

# Step 5: Perform Inference -- Style Transfer
# Select style image
# Select the style id (the mapping between id and style image filename are defined here) We assume that you have downloaded the dataset and placed them in ./data/STYLE_TYPE/images/train/. If not, the style images will not be shown below.

style_id = 26  # style_type에 원하는 id 입력 여러 개 중에 우리가 쓸 이미지는 미리 지정해놓기

# try to load the style image
stylename = list(exstyles.keys())[style_id]
stylepath = os.path.join(DATA_DIR, style_type, 'images/train',
                         stylename)  # DualStyleGAN/data/style_type/images/train/stylename
print('loading %s' % stylepath)  # style 경로 로딩 중
if os.path.exists(stylepath):  # stylepath에 값이 있으면 실행
    S = load_image(stylepath)  # stylepath에 있는 이미지를 S에 저장
else:
    print('%s is not found' % stylename)

# Style transfer with and without color preservation
with torch.no_grad():
    img_rec, instyle = encoder(I, randomize_noise=False, return_latents=True,
                               z_plus_latent=True, return_z_plus_latent=True, resize=False)
    img_rec = torch.clamp(img_rec.detach(), -1, 1)

    latent = torch.tensor(exstyles[stylename]).repeat(2, 1, 1).to(device)
    # latent[0] for both color and structrue transfer and latent[1] for only structrue transfer
    latent[1, 7:18] = instyle[0, 7:18]
    exstyle = generator.generator.style(latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])).reshape(
        latent.shape)

    img_gen, _ = generator([instyle.repeat(2, 1, 1)], exstyle, z_plus_latent=True,
                           truncation=0.7, truncation_latent=0, use_res=True, interp_weights=[0.6] * 7 + [1] * 11)
    img_gen = torch.clamp(img_gen.detach(), -1, 1)
    # # deactivate color-related layers by setting w_c = 0
    # img_gen2, _ = generator([instyle], exstyle[0:1], z_plus_latent=True,
    #                         truncation=0.7, truncation_latent=0, use_res=True, interp_weights=[0.6] * 7 + [0] * 11)
    # img_gen2 = torch.clamp(img_gen2.detach(), -1, 1)

print('Generate images successfully!')  # 이미지 생성 완료

# 결과 이미지 저장 output 디렉터리 없으면 생성하기
if not os.path.exists(os.path.join(os.getcwd(), 'output')):  # ModelServer/DualStyleGAN/checkpoint/style_type의 디렉터리가 없으면 실행
    os.makedirs(os.path.join(os.getcwd(), 'output'))  # ModelServer/DualStyleGAN/checkpoint/style_type으로 디렉터리 생성

save_name = '%s_transfer_%d_%s' % (style_type, style_id, 'test')
save_image(img_gen[0].cpu(), os.path.join('./output/', save_name + '.jpg'))  # 결과 이미지

print('Save images successfully!')  # 이미지 저장 완료

for i in range(6):  # change weights of structure codes, change weights of color codes
    w = [i / 5.0] * 7 + [i / 5.0] * 11

    img_gen, _ = generator([instyle], exstyle[0:1], z_plus_latent=True,
                           truncation=0.7, truncation_latent=0, use_res=True, interp_weights=w)
    img_gen = torch.clamp(img_gen.detach(), -1, 1)
    save_image(img_gen[0].cpu(), os.path.join('./output/', save_name + '%d' + '.jpg') % i)  # 결과 이미지

print('Done!')

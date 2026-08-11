# 로컬 셋업 가이드 (사내 서버, 2× RTX A6000)

2026-08-10 기준, `dayo@` 서버에서 VLMaterial을 돌리기 위해 실제로 수행한 세팅 기록입니다.
공식 README의 설치 절차를 따르되, **사내망(프록시) 환경과 이 머신 특성 때문에 달라진 부분**을 중심으로 정리했습니다.

## 머신 사양

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX A6000 48GB × 2 |
| Driver / CUDA | 570.211.01 / CUDA 12.8 (시스템에 nvcc 없음) |
| OS | Ubuntu (Linux 6.8.0-124-generic) |
| Conda | `~/miniconda3` |
| 모델 파티션 | `/mnt/models` (1.9T, ComfyUI 모델 디렉토리와 공유) |

README는 학습에 48GB VRAM 이상을 권장하므로 A6000 2장으로 학습/추론 모두 가능합니다.
(단, 기본 스크립트는 8× H100 기준이라 배치 사이즈를 줄였습니다 — 아래 참고.)

## 디렉토리 배치

큰 파일(데이터셋, 가중치)은 전부 모델 파티션 `/mnt/models`에 두고 repo에는 심링크만 둡니다.

```
/mnt/models/
├── huggingface/                        # HF 캐시 (HF_HOME) — 베이스 VLM 등
└── vlmaterial/
    ├── material_dataset_filtered/      # 데이터셋 (압축 해제)
    └── checkpoints_pretrained/         # 사전학습 LoRA 가중치 (압축 해제)

/home/dayo/VLMaterial/
├── material_dataset_filtered           -> /mnt/models/vlmaterial/material_dataset_filtered
└── llava_hf/checkpoints_pretrained     -> /mnt/models/vlmaterial/checkpoints_pretrained
```

`/mnt/models` 밑 디렉토리 생성은 root 소유라 sudo가 필요합니다. (이번 세팅에서는
docker 그룹 권한으로 `docker run -v /mnt/models:/m alpine chown ...`으로 처리했음.)

## 1. Conda 환경

```bash
conda env create -f environment.yml   # ⚠️ pip 단계에서 deepspeed 빌드 실패함 (아래 참고)
conda activate vlmaterial
```

### 삽질 1: deepspeed 빌드 실패

`environment.yml`의 pip 단계에서 `deepspeed==0.15.2`가 소스 빌드되는데 두 가지 이유로 실패:

1. **pip build isolation** 때문에 빌드 환경에서 torch가 안 보임
   → `--no-build-isolation`으로 설치해야 함.
2. **시스템에 nvcc가 없어서** `CUDA_HOME does not exist` 에러
   → conda로 CUDA 11.8 툴체인을 env 안에 설치하고 `CUDA_HOME`을 env로 지정.

해결 순서 (conda 패키지들은 `env create`에서 이미 설치된 상태):

```bash
conda activate vlmaterial
# environment.yml의 pip 목록 중 deepspeed 제외하고 먼저 설치
pip install accelerate==1.0.1 ... transformers==4.45.2   # (전체 목록은 environment.yml 참고)
# nvcc 설치 후 deepspeed
conda install -c nvidia cuda-nvcc=11.8 cuda-toolkit=11.8
export CUDA_HOME=$CONDA_PREFIX
pip install deepspeed==0.15.2 --no-build-isolation
```

> 학습 시에도 deepspeed가 CUDA op을 JIT 컴파일할 수 있으므로 `CUDA_HOME=$CONDA_PREFIX`를
> 유지해야 합니다. `scripts/peft_sllm_p10.sh`에 export 해두었습니다.

### flash-attn

README대로 prebuilt wheel 설치 (torch 2.4.1 + cu118 + CXX11_ABI=FALSE 확인 완료):

```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-*.whl --no-build-isolation
```

검증:

```bash
python -c "import flash_attn, deepspeed, transformers, peft, accelerate, lpips, openai, tensorboardX; print('OK')"
```

## 2. Blender (Infinigen)

Infinigen의 `install.sh` 전체를 돌릴 필요가 없습니다. VLMaterial이 Infinigen에서 실제로 쓰는 것은:

- `infinigen/blender/blender` 실행 파일 (스크립트들의 기본 `--blender_path`)
- `infinigen/worldgen/nodes/node_transpiler/` (순수 Python 모듈, 빌드 불필요)

terrain/NURBS 컴파일, infinigen requirements.txt 전체 설치는 material 파이프라인에 안 쓰이므로 생략했습니다.
실제 수행한 것:

```bash
cd infinigen
wget https://download.blender.org/release/Blender3.3/blender-3.3.1-linux-x64.tar.xz -O blender.tar.xz
tar -xf blender.tar.xz && mv blender-3.3.1-linux-x64 blender && rm blender.tar.xz

# Blender 내장 Python에 스크립트가 요구하는 패키지 설치 (numpy는 내장됨)
blender/3.3/python/bin/python3.10 -m ensurepip
blender/3.3/python/bin/python3.10 -m pip install Pillow tqdm
```

검증 (transpiler import 스모크 테스트):

```bash
infinigen/blender/blender -b --python-expr "
import sys, os
sys.path.append(os.path.join(os.getcwd(), 'infinigen', 'worldgen'))
from nodes.node_transpiler import transpiler_blender
print('TRANSPILER_IMPORT_OK')"
```

## 3. 모델/데이터 다운로드 — 사내망 이슈 ⚠️

### HuggingFace: Artifactory 미러 + 토큰 버그

이 서버는 `/etc/environment`에서 `HF_ENDPOINT`가 사내 Artifactory 미러
(`bart.sec.samsung.net/.../huggingface-remote`)로 잡혀 있습니다. 그런데 **`HF_TOKEN` 줄 뒤에
인라인 주석(`# dayo's token`)이 붙어 있어서 pam_env가 닫는 따옴표+공백까지 값에 포함**시키고,
그 결과 모든 HF 요청이 401로 실패합니다.

- 임시 보정: `~/.zshrc`에서 깨끗한 값으로 `export HF_TOKEN=...` 재정의 (적용해둠)
- 근본 해결: `/etc/environment`에서 해당 줄의 인라인 주석을 제거 (sudo 필요)

### 삽질 2: Artifactory 미러는 대용량(LFS) 파일에서 타임아웃

Artifactory 미러 경유 다운로드는 **작은 파일(config, tokenizer 등)은 성공하지만
safetensors 샤드(각 ~5GB)는 매번 ~22분 후 실패**합니다 (미러가 업스트림 캐싱 중
게이트웨이 타임아웃으로 추정, 30회 재시도해도 동일). `huggingface.co` 직접 접근은
프록시에서 허용되지만 기본 CDN(`us.aws.cdn.hf.co`)은 행이 걸립니다.

**해결: Xet 전송 백엔드** — Xet의 CAS 서버(`cas-server.xethub.hf.co`,
`transfer.xethub.hf.co`)는 프록시를 통과합니다. `hf_xet`을 설치하고 미러를 우회하면
대용량 파일도 정상 다운로드됩니다:

```bash
pip install hf_xet
export HF_ENDPOINT=https://huggingface.co   # Artifactory 미러 우회 (이 세션에서만)
export HF_TOKEN=                            # Artifactory 토큰을 hf.co에 보내지 않도록 비움
export HF_HOME=/mnt/models/huggingface      # ~/.zshrc에 추가해둠
python -c "from huggingface_hub import snapshot_download; snapshot_download('llava-hf/llama3-llava-next-8b-hf')"
```

> 평상시(작은 모델/파일)는 사내 미러를 그대로 쓰고, 대용량 모델을 받을 때만
> 위처럼 `HF_ENDPOINT`를 일시적으로 덮어쓰는 것을 권장합니다.

### Google Drive: 프록시에서 차단 🚫

데이터셋(`material_dataset_filtered.zip`)과 사전학습 가중치(`checkpoints_pretrained.zip`)는
Google Drive 배포인데, 사내 프록시가 `drive.google.com`을 차단합니다 ("Secure Data Center Notice").
`gdown`, `drive.usercontent.google.com` 직접 접근 모두 실패.

**→ 사외망(개인 PC 등)에서 받아서 서버로 옮겨야 합니다:**

- 데이터셋: <https://drive.google.com/file/d/1k85IeCpKsduPbSycWLgnw5c1NxM_ENl4/view>
- 가중치: <https://drive.google.com/file/d/1mx66RcGFZIN2f-vDvU1NWtNRBvwHZyDx/view>

옮긴 후:

```bash
unzip -q material_dataset_filtered.zip -d /mnt/models/vlmaterial/
unzip -q checkpoints_pretrained.zip -d /mnt/models/vlmaterial/
# repo의 심링크가 이미 위 경로를 가리키므로 추가 작업 불필요. 확인:
ls ~/VLMaterial/material_dataset_filtered/infinigen/wood/blender_full.py
```

## 4. X11 / OpenGL (렌더링 단계에서만 필요) ⚠️

EEVEE 렌더링(추론 2단계, 데이터셋 필터링/증강)은 OpenGL 컨텍스트가 필요합니다.
이 서버는 gdm의 X 서버(`:0`)가 **다른 계정(cdoax) 소유**라 dayo 계정으로 붙을 수 없습니다.
SSH X11 포워딩(`localhost:10.0`)은 로컬 렌더링이라 부적합.

전용 X 서버를 하나 띄워야 합니다 (sudo 필요, 최초 1회):

```bash
sudo nvidia-xconfig --enable-all-gpus   # /etc/X11/xorg.conf 생성 (기존 :0과 충돌 시 주의)
sudo X :1 &
export DISPLAY=:1
```

스크립트들의 `--display_id`는 `1`로 맞춰두었습니다. 학습(`peft_sllm_p10.sh`)과
추론 1단계(generation)는 X 없이 동작합니다.

## 5. 스크립트 변경 사항 (이 머신 기준)

`llava_hf/scripts/{peft,eval,render}_sllm_p10.sh`:

- `ROOT_DIR=/home/dayo/VLMaterial`
- GPU 2장: `DEVICE_IDS="0,1"`, `--num_processes 2`, `--device_id 0 1`
- A6000 48GB (H100 80GB 대비 축소): train batch 4→1, `--gradient_accumulation_steps` 1→4
  (유효 배치 유지: 8×4×1 = 32 vs 2×1×4 = 8 — 필요하면 accum을 16으로 올려 32 유지 가능)
- `--model_path`를 사전학습 가중치(`checkpoints_pretrained`)로 변경
- `HF_HOME`, `CUDA_HOME` export 추가
- `--display_id 1` (전용 X 서버 기준)

## 실행 순서 요약

```bash
conda activate vlmaterial
cd ~/VLMaterial/llava_hf

# (선행조건: 3번의 zip 2개 반입 + 4번의 X 서버)
bash scripts/eval_sllm_p10.sh     # 1) 후보 프로그램 생성 (GPU만 필요)
bash scripts/render_sllm_p10.sh   # 2) 검증 + 렌더링 (X11 필요)
bash scripts/peft_sllm_p10.sh     # (옵션) 직접 파인튜닝
```

## 남은 작업 체크리스트

- [ ] 사외망에서 데이터셋/가중치 zip 2개 반입 → `/mnt/models/vlmaterial/`에 해제
- [ ] `sudo X :1` 로 렌더링용 X 서버 기동 (renders/필터링 단계 전)
- [ ] (권장) `/etc/environment`의 `HF_TOKEN` 인라인 주석 제거
- [ ] 데이터 증강을 돌릴 경우 `data_scripts/gen_programs_llm_async.py`에 OpenAI 호환 API 자격증명 필요
      (사내망에서 api.openai.com 접근 가능 여부 별도 확인 필요)

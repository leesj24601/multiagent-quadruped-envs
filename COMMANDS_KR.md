# Multi-agent Quadruped Environment (MQE) 명령어 가이드

이 문서는 MQE 프로젝트에서 사용할 수 있는 주요 명령어들을 한국어로 정리한 것입니다.

## 1. 설치 (Installation)

환경을 설정하고 필요한 패키지를 설치하는 명령어입니다.

### 가상 환경 생성
Python 3.8 환경을 권장합니다.
```bash
conda create -n mqe python=3.8
```

### PyTorch 및 Isaac Gym 설치
PyTorch 설치 (CUDA 11.3 기준):
```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Isaac Gym 설치 (Preview 4 버전 필요):
```bash
# Isaac Gym 패키지 압축 해제 후
cd isaacgym/python && pip install -e .
```

### MQE 설치
이 리포지토리의 루트 디렉토리에서 실행하세요.
```bash
pip install -e .
```

---

## 2. 실행 (Usage)

### 기본 실행 (Basic Run)
기본 설정으로 환경을 실행하여 테스트합니다. 실행할 태스크는 `test.py` 파일 내에서 수정할 수 있습니다.
```bash
python ./test.py
```

### 학습 (Training)
OpenRL을 사용하여 협동 태스크를 학습합니다.
```bash
python ./openrl_ws/train.py --algo ALGO_NAME --task TASK_NAME
```
*   `--num_envs NUM_ENVS`: 병렬 시뮬레이션 환경 수 지정
*   `--train_timesteps NUM_STEPS`: 학습 단계 수 지정
*   `--headless`: 화면 렌더링 없이 실행 (서버 등에서 유용)
*   `--use_wandb`: WanDB 사용
*   `--use_tensorboard`: TensorBoard 사용

### 평가 (Evaluation)
학습된 정책(Policy)을 평가합니다.
```bash
python ./openrl_ws/test.py --algo ALGO_NAME --task TASK_NAME --checkpoint /PATH/TO/CHECKPOINT
```
*   `--record_video`: 비디오 녹화

---

## 3. 사용 가능한 태스크 (Available Tasks)

`--task` 옵션이나 `test.py`의 `task_name` 변수에 사용할 수 있는 태스크 목록입니다.

### 협동 태스크 (Collaborative Tasks)
*   `go1gate`: 두 로봇이 좁은 문을 순차적으로 통과
*   `go1seesaw`: 두 로봇이 시소를 타고 플랫폼으로 이동
*   `go1sheep-easy`: 두 로봇이 양 한 마리를 문으로 몰기
*   `go1sheep-hard`: 두 로봇이 양 아홉 마리를 문으로 몰기
*   `go1pushbox`: 두 로봇이 무거운 상자를 문으로 밀기
*   `go1football-defender`: 두 로봇이 수비수를 피해 골 넣기

### 경쟁 태스크 (Competitive Tasks)
*   `go1tug`: 두 로봇이 서로 반대 방향으로 원통 밀기
*   `go1revolvingdoor`: 회전문을 통과하여 반대편으로 이동
*   `go1bridge`: 좁은 다리를 건너 반대편으로 이동 (상대방 밀치기 가능)
*   `go1bridgewrestling`: 스모 경기장에서 상대를 밀어내거나 넘어뜨리기
*   `go1football-1vs1`: 1대1 축구
*   `go1football-2vs2`: 2대2 축구

---

## 4. 문제 해결 (Troubleshooting)

*   `ImportError: libpython3.8m.so.1.0...`: `LD_LIBRARY_PATH` 환경변수 설정 필요.
    ```bash
    export LD_LIBRARY_PATH=/PATH/TO/CONDA/envs/mqe/lib
    ```
*   `AttributeError: module 'numpy' has no attribute 'float'`: numpy 버전 호환성 문제. 1.20.3 버전으로 재설치 권장.
    ```bash
    pip install numpy==1.20.3
    ```

# 프로젝트 구조 메모

## 새 태스크를 만들려면 필요한 파일

태스크 하나를 만들기 위해 필요한 파일 세트:

```
mqe/envs/configs/go1_<태스크명>_config.py   ← 환경 설정 (지형, 로봇, 보상 스케일 등)
mqe/envs/wrappers/go1_<태스크명>_wrapper.py ← 관측, 행동, 보상 함수 정의
resources/objects/<오브젝트>.urdf            ← 필요한 경우 새 물체
```

그리고 아래 파일들에 등록해줘야 함:

```
mqe/envs/__init__.py        ← wrapper import 추가
mqe/utils/task_registry.py  ← 태스크 이름 → config/wrapper 매핑 추가
mqe/envs/utils.py           ← 태스크 설정 연결
```

---

## 내가 새로 추가한 태스크

### go1pushball
- **파일**: `go1_pushball_config.py`, `go1_pushball_wrapper.py`, `ball_heavy.urdf`, `ball_large.urdf`
- **태스크**: 2개 에이전트가 협력해서 공을 벽의 구멍으로 밀어넣기
- **보상 구조**:
  - 공이 목표(구멍)에 가까워질수록 양의 보상 (shared)
  - 에이전트가 공에 가까워질수록 보상 (individual)
  - 공에 접촉 시 보상 (individual)
  - 성공(공이 구멍 통과) 시 큰 보상 + 에피소드 종료
- **지형**: init 블록 + hole_wall 블록

### go1pushbox_light / go1pushbox_medium
- **파일**: `go1_pushbox_light_config.py`, `go1_pushbox_medium_config.py`, `box_light.urdf`, `box_medium.urdf`
- **태스크**: 기존 go1pushbox의 변형. 박스 무게만 다름
- **구조**: `Go1PushboxCfg`를 상속해서 URDF와 보상 스케일만 변경

### go1gatewithbutton
- **파일**: `go1_gate_with_button_config.py`, `go1_gate_with_button_wrapper.py`, `gate.urdf`
- **태스크**: 한 에이전트가 버튼을 밟아 게이트를 열면, 다른 에이전트가 통과
- **보상 구조**:
  - 버튼 누르고 있으면 지속 보상 (shared)
  - 에이전트 간 거리 너무 가까우면 패널티 (역할 분리 유도)
  - 에이전트가 게이트(x > 4.0) 통과 시 성공 보상 + 에피소드 종료
- **게임 로직**: wrapper의 `step()`에서 버튼 감지 → 게이트 Z좌표 조정
- **지형**: init + gate + plane + wall 블록
- **주요 config**: `cfg.game.button_pos`, `button_radius`, `gate_open_height`, `gate_closed_height`

---

## 실행 방법

```bash
# 테스트 (test.py에서 task 이름 수정)
python ./test.py

# 학습
python ./openrl_ws/train.py --algo MAPPO --task go1pushball
python ./openrl_ws/train.py --algo MAPPO --task go1gatewithbutton
python ./openrl_ws/train.py --algo MAPPO --task go1pushbox_light
python ./openrl_ws/train.py --algo MAPPO --task go1pushbox_medium
```

---

## Git 원격 저장소

- `origin` → 내 레포: https://github.com/leesj24601/multiagent-quadruped-envs
- `upstream` → 원본 레포: https://github.com/ziyanx02/multiagent-quadruped-environment

원본 업데이트 받으려면:
```bash
git fetch upstream
git merge upstream/main
```

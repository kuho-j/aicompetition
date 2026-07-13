# 실행 환경
본 프로젝트는 uv 환경에서 실행했습니다.

- Python : 3.12
- OS : Linux x86_64
- PyTorch : 2.5.x, CUDA 12.4 wheel 사용
- 주요 패키지 : torch, torchvision, scikit-learn

프로젝트를 다운 받은 뒤, `uv sync`를 입력해서 필요한 패키지를 설치할 수 있습니다.

```bash
uv sync
```

# 프로젝트 실행 방법
`main.py`를 실행시키면 task를 수행합니다.
`main.py`는 다음 6가지 인자를 입력해야 합니다.

| 인자 | 설명 |
| :-: | :-: |
| --weights | 모델에 사용할 가중치 파일입니다. |
| --cam1 | 첫 번째 비디오 파일의 경로입니다. |
| --cam2 | 두 번째 비디오 파일의 경로입니다. |
| --cam3 | 세 번째 비디오 파일의 경로입니다. |
| --cam4 | 네 번째 비디오 파일의 경로입니다. |
| --cam5 | 다섯 번째 비디오 파일의 경로입니다. |

```bash
uv run python --weights path/to/model.pt --cam1 ... --cam2 ... --cam3 ... --cam4 ... --cam5 ...
```

결과는 현재 디렉토리의 `result.csv`에 저장됩니다.
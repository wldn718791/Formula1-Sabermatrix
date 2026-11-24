# F1 STATIZ — 세이버메트릭스 F1 드라이버 랭킹 사이트

단순 랩타임과 같이 고전적인 스텟에서 벗어나 새로운 관점으로 드라이버를 바라볼 수 있도록 만든 F1판 세이버메트릭스 사이트 입니다.

## 파일 구성

- `index.html`: `./data/f1_war_rankings.csv`를 자동 로드하는 스탯티즈 스타일 웹 대시보드
- `update_incremental_seasons.py`: 2024~2025 아카이브 재사용 + 2026 신규 라운드 증분 수집 스크립트
- `jolpica_f1_loader.py` / `fastf1_enricher.py` / `f1_multilevel_prototype.py` / `run_f1_sabermetrics_pipeline.py`: 데이터 수집·보정·다층 모형 코어 모듈
- `.github/workflows/update_f1_stats.yml`: 매주 월요일 오전 09:00(KST) 자동 갱신 및 수동 실행 워크플로
- `data/f1_war_rankings.csv`: 대시보드가 자동으로 읽는 최종 랭킹 파일 (초기 데모 데이터 포함)

## 3단계 배포 방법

1. **GitHub 저장소에 업로드**: 이 폴더의 모든 파일(숨김 폴더 `.github/` 포함)을 새 공개(Public) GitHub 저장소의 `main` 브랜치에 푸시합니다.
2. **GitHub Pages 켜기**: 저장소 `Settings → Pages`에서 Source를 `Deploy from a branch`, Branch를 `main` / `/ (root)`로 저장하면 몇 분 안에 사이트가 열립니다.
3. **최초 실제 데이터 갱신**: `Actions → Update F1 STATIZ Rankings → Run workflow` 버튼을 누르면 2024~2025 아카이브와 2026 최신 기록이 수집되어 `data/f1_war_rankings.csv`가 자동 갱신됩니다. (선택 사항: `data/salaries.csv`를 추가하면 연봉 가성비 컬럼이 자동 활성화됩니다.)

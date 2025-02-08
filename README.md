# Continual Retrieval
- Packages
- Data
- Components 
- Contribution


# Packages
```
.
├── README.md
├── data
│    ├── model
│    ├── rankings
│    └── sesions
└── src
    ├── main.py
    ├── buffer
    │   ├── __init__.py
    │   ├── buffer_utils.py 
    │   ├── gss_greedy_update.py 
    │   ├── mir_retrieve.py 
    │   ├── random_retrieve.py 
    │   ├── reservoir_update.py 
    │   ├── sampler.py
    ├── cluster
    │   ├── __init__.py
    │   ├── management.py 
    │   ├── prototype.py 
    │   ├── retriever.py 
    │   └── sampler.py
    ├── data
    │   ├── __init__.py
    │   ├── generate.py
    │   ├── loader.py
    │   └── preprocess.py
    ├── functions
    │   ├── __init__.py
    │   ├── evaluate.py
    │   ├── loss.py
    │   ├── retriever.py
    │   ├── similarities.py
    │   └── utils.py
    └── pipeline
        ├── __init__.py
        ├── bm25_ranking.py
        ├── ground_truth_train_ranking.py
        ├── proposal_train_ranking.py
        ├── random_train_ranking.py
        ├── find_optimized_cluster.py
        └── memory_based_train_ranking.py
```
# Data
- model
  - checkpoint 경로
- rankings
  - evaluate.py에서 읽어가는 pipeline 랭킹 결과물
- sessions
  - .local 공용 폴더의 /raw/idea_validation 이하

# Components
### buffer
- 쿼리버퍼 내 샘플 선정 전략(l2r, bm25, ER, MIR, GSS, OCS)

### cluster
- management
  - 클러스터 초기화 및 평가(kmeans++)
  - 클러스터 프로토타입, 인스턴스 매핑, 통계 관리 
- prototype
  - RP-LSH
- retriever
  - 클러스터 내 리트리버
- sampler
  - 클러스터 내 샘플링

### data
- loader
  - jsonl 등 입출력
- generate
  - merge, sessioning, train/val/test split 등
- preprocess  
  - 전체 데이터 인코딩 등

### functions
- evaluate
  - 전체 or cluster에서 쿼리에 대한 정답문서 조회
- loss
  - InfoNCELoss
- retriever
  - 클러스터 없는(전체문서) 리트리버
- similarties
  - cosine, term score 등
- utils
  - 시각화, collection 등

### pipeline, main
각종 실험들, 엔트리 포인트


# Contribution
- black으로 린팅
- python main.py --exp=proposal 와 같이 실행
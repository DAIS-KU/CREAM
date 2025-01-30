# Continual Retrieval
- Packages
- Components 
- Flow
- Contribution


# Packages
```
.
├── README.md
├── data
├── model
├── rankings
└── src
    ├── cluster
    │   ├── __init__.py
    │   ├── management.py 
    │   └── prototype.py
    ├── data
    │   ├── __init__.py
    │   ├── generate.py
    │   ├── loader.py
    │   └── preprocess.py
    ├── functions
    │   ├── __init__.py
    │   ├── retriever.py
    │   ├── sampler.py
    │   ├── similarities.py
    │   └── utils.py
    ├── train.py
    ├── evaluate.py
    └── unit_test.py

```

# Components
### cluster
- management
  - 클러스터 초기화 및 평가(kmeans++)
  - 클러스터 프로토타입, 인스턴스 매핑, 통계 관리 
- prototype
  - RP-LSH

### data
- loader
  - jsonl 등 입출력
- generate
  - merge, sessioning, train/val/test split 등
- preprocess  
  - 전체 데이터 인코딩 등

### functions
- retriever
  - 전체 or cluster에서 쿼리에 대한 정답문서 조회
- sampler
  - p/n 선정 전략별(l2r, bm25....) 쿼리 학습 샘플 선정
- similarties
  - cosine, term score 등
- utils
  - 시각화, collection 등


### main
- train
- evaluate
- unit test


# Contribution
- black으로 린팅
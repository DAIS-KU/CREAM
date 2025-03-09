# compatible 옵션 사용법
- buffer_emb.pkl 과거 임베딩
  - https://github.com/caiyinqiong/L-2R/blob/022f1282cd5bdf42c29d0f006e5c1b77c2c5c724/run_example.sh#L94
  - L2R에서는 매 세션을 각각 호출했덙 것 같음(수정 필요?)
- buffer.pkl 쿼리 별 문서 아이디, 리저버 샘플 갯수 파라미터
- buffer.replace() 역할은?
# compatible 옵션 사용법
- buffer_emb.pkl 과거 임베딩
  - https://github.com/caiyinqiong/L-2R/blob/022f1282cd5bdf42c29d0f006e5c1b77c2c5c724/run_example.sh#L94
  - L2R에서는 매 세션을 각각 호출했덙 것 같음(수정 필요?)
- buffer.pkl 쿼리 별 문서 아이디, 리저버 샘플 갯수 파라미터
- buffer.replace() 역할은?
  - buffer.update_old_embs랑 동일


# identity
- 재사용 임베딩인 원소 인덱스 리스트 왜 텐서로 사용하는지 확인 필요 
- 매 세션 학습은 old로, 저장은 new emb로?
- 그것도 그런데 새로운 스트림의 문서도 기존 임베딩을 쓸거라서 최초 모든 문서 임베딩 필요(!)
  - https://github.com/caiyinqiong/L-2R/blob/main/src/tevatron/buffer/generate_buffer_emb.py
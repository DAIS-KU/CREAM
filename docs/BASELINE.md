베이스라인 코드에 대한 설명입니다.
/src/pipeline/test_buffer.py를 참고해주세요.

# Buffer
- 객체 생성 시 Argument path에서 buffer.pkl값을 읽어오거나 메모리 상 컬렉션을 초기화 합니다.
- trainer에서 모델 저장 시 buffer.save()를 함께 호출합니다.

# DenseModel
- L2R과 베이스라인들은 기본 버트 대신 위 모델을 사용합니다.
- identity, old_emb는 compatible 설정일 떄만 사용되는 값으로, 재활용할 문서 아이디와 예전 임베딩입니다.
    - trainer의 _prepare_inputs에서 메모리 조회 및 업데이트하여 prepare에 identity, old_emb를 추가해 반환해줍니다.
- 학습 시 해당 반환값으로 모델을 업데이트합니다.


# Input형태
- Dataset -> Collator -> DataLoader 를 거쳐 trainer의 _prepare_inputs의 inputs값으로 들어갑니다.
  - Dataset에서는 input_ids만 포함된 dict
    - positive_passage_no_shuffle=False "always use the first positive passage" (negative_passage_no_shuffle=False)
    - 총 8개 샘플학습
    - 최초 세션에서는 positive, negative 모두 주어져야함(아마 다른 문서의 positive)
  - Collator에서 pad, attn_mask 추가
  - DataLoader에서 미니배치구성
- inputs = qid_lst, did_lst, q_lst, d_lst
  - qid_lst, did_lst는 쿼리, 문서 아이디 리스트
  - q_lst, d_lst는 모델입력값 (Dict[str, Tensor])로 input_ids, attention_mask입니다


# Discussion
- 작성한 베이스라인 파이프라인상 개별 쿼리로 각 샘플링 후 loss업데이트 시에만 배치 작업하고 있음, 배치 형태 입출력 파악필요한가?
- huggingface Dataset, Collator, DataLoader, Trianer를 사용해야할까? tevatron library를 사용해야할까?
  - 일단 직접 구현 후 필요시 적용
  - functions.load_collated_data(): Dataset+Collator+DataLoader의 결과인 inputs 생성하는 함수(prepare_inputs의 입력)
  - functions.prepare_inputs(): TevatronTrainer의 _prepare_inputs() 메소드.
  - memory_based_train_ranking.py에서는 쿼리가 아니라 functions.prepare_inputs를 순회하여 학습한다.
- 구현되어 있는 Loss를 우리도 사용하게 수정?
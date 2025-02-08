# https://github.com/caiyinqiong/L-2R/tree/main/src/tevatron/buffer
def memory_based_sampling(query, method):
    if method == "ER":
        return sample_by_er(query)
    elif method == "MIR":
        return sample_by_mir(query)
    elif method == "GSS":
        pass
    elif method == "OCS":
        pass
    elif method == "L2R":
        pass
    else:
        raise ValueError(f"Unsupported memory_based_sampling method {method}")
    
def memory_update(query, docs, method):
    if method == "ER":
        update_by_er(query, docs)
    elif method == "MIR":
        update_by_mir(query, docs)
    elif method == "GSS":
        pass
    elif method == "OCS":
        pass
    elif method == "L2R":
        pass
    else:
        raise ValueError(f"Unsupported memory_update method {method}")


def sample_by_er(query):
    """
    메모리에서 재생할 데이터 선택에 랜덤 샘플링 사용,
    메모리에 남길 데이터 선택에 리저버 샘플링 사용
    """
    pass


def update_by_er(query, docs):
    """
    메모리에서 재생할 데이터 선택에 랜덤 샘플링 사용,
    메모리에 남길 데이터 선택에 리저버 샘플링 사용
    """
    pass


def sample_by_mir(query):
    """
    새로운 데이터로 학습된 업데이트된 모델에 대한 손실 증가량을 기준 재생 샘플을 선택,
    메모리 업데이트에 리저버 샘플링 사용
    """
    pass


def update_by_mir(query, docs):
    """
    새로운 데이터로 학습된 업데이트된 모델에 대한 손실 증가량을 기준 재생 샘플을 선택,
    메모리 업데이트에 리저버 샘플링 사용
    """
    pass
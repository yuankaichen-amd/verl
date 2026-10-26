import pytest
import torch

try:
    from verl.models.mcore import util
except OSError as error:
    pytest.skip(f"Megatron runtime is unavailable on this host: {error}", allow_module_level=True)


class _CP:
    def __init__(self):
        self.rank = 0
        self.size = 1

    def tp(self):
        return 1

    def cp(self):
        return self.size

    def rank_of(self):
        return self.rank


def test_cp2_zigzag_thd_preserves_variable_length_tokens(monkeypatch):
    state = _CP()
    monkeypatch.setattr(util.mpu, "get_tensor_model_parallel_world_size", state.tp)
    monkeypatch.setattr(util.mpu, "get_context_parallel_world_size", state.cp)
    monkeypatch.setattr(util.mpu, "get_context_parallel_rank", state.rank_of)

    sequences = [torch.arange(1, 7), torch.arange(100, 113)]
    packed = torch.nested.nested_tensor(sequences, layout=torch.jagged)
    state.size = 2
    local = []
    params = None
    for rank in (0, 1):
        state.rank = rank
        shard, params, _positions = util.preprocess_thd_engine(packed, cp_layout="zigzag")
        local.append(shard)
        assert params.qkv_format == "thd"

    restored = []
    for seq_index, original in enumerate(sequences):
        width = int((params.cu_seqlens_q_padded[seq_index + 1] - params.cu_seqlens_q_padded[seq_index]).item())
        half = width // state.size // 2
        pieces = []
        for shard in local:
            start = int(params.cu_seqlens_q_padded[seq_index].item()) // state.size
            pieces.append(shard[0, start : start + half])
        for shard in reversed(local):
            start = int(params.cu_seqlens_q_padded[seq_index].item()) // state.size
            pieces.append(shard[0, start + half : start + 2 * half])
        restored.append(torch.cat(pieces)[: original.numel()])
        torch.testing.assert_close(restored[-1], original)

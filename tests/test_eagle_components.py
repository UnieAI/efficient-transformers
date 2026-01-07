
import sys
import os
import torch
import numpy as np
import pytest

# Ensure we can import from the project root and examples
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../examples/performance/speculative_decoding")))

from QEfficient.transformers.spd.eagle import EagleConfig, EagleHead, EagleDraftLoop
from eagle_utils import EagleFeatureCache, normalize_token_ids, normalize_hidden_states

@pytest.fixture
def eagle_config():
    return EagleConfig(vocab_size=100, hidden_size=32, num_attention_heads=4)

@pytest.fixture
def eagle_head(eagle_config):
    return EagleHead(eagle_config, hidden_size=32)

def test_normalize_token_ids():
    # Test 1D input
    arr = np.array([1, 2, 3])
    out = normalize_token_ids(arr)
    assert out.ndim == 2
    assert out.shape == (3, 1)

    # Test 2D input
    arr = np.array([[1, 2], [3, 4]])
    out = normalize_token_ids(arr)
    assert out.shape == (2, 2)

    # Test error
    with pytest.raises(ValueError):
        normalize_token_ids(np.zeros((1, 1, 1)))

def test_feature_cache():
    cache = EagleFeatureCache(max_len=3)
    
    # Batch=1, Seq=1
    token = np.array([[1]])
    feature = np.zeros((1, 1, 32))
    
    # Append 1
    cache.append(token, feature)
    t, f = cache.window()
    assert t.shape == (1, 1)
    
    # Append 2
    cache.append(token, feature)
    t, f = cache.window()
    assert t.shape == (1, 2)
    
    # Append 3
    cache.append(token, feature)
    t, f = cache.window()
    assert t.shape == (1, 3)
    
    # Append 4 (Should pop 1, keep len 3)
    cache.append(token, feature)
    t, f = cache.window()
    assert t.shape == (1, 3)

def test_eagle_head_shapes(eagle_head):
    batch = 2
    seq = 5
    hidden = 32
    
    input_ids = torch.randint(0, 100, (batch, seq))
    history_features = torch.randn(batch, seq, hidden)
    
    logits, new_features = eagle_head(input_ids, history_features)
    
    assert logits.shape == (batch, seq, 100)
    assert new_features.shape == (batch, seq, hidden)

def test_eagle_head_kv_cache(eagle_head):
    batch = 1
    hidden = 32
    head_dim = 8 # 32/4
    
    input_ids = torch.randint(0, 100, (batch, 1))
    history_features = torch.randn(batch, 1, hidden)
    
    # Initial step (past_key=None triggers masking for seq_len)
    # But here we pass explicit None
    past_key = torch.zeros(batch, 4, 1, head_dim)
    past_value = torch.zeros(batch, 4, 1, head_dim)
    
    logits, new_features, present_key, present_value = eagle_head(
        input_ids, 
        history_features, 
        past_key=past_key, 
        past_value=past_value
    )
    
    # present_key should be length 1 (past) + 1 (current) = 2
    assert present_key.shape == (batch, 4, 2, head_dim)
    assert present_value.shape == (batch, 4, 2, head_dim)

def test_eagle_draft_loop(eagle_head):
    num_steps = 3
    model = EagleDraftLoop(eagle_head, num_steps=num_steps, use_cache=False)
    
    batch = 1
    hidden = 32
    input_ids = torch.randint(0, 100, (batch, 1))
    history_features = torch.randn(batch, 1, hidden)
    
    tokens, features = model(input_ids, history_features)
    
    # Output should be concatenated across steps?
    # EagleDraftLoop collects steps.
    # tokens -> [batch, steps]
    # features -> [batch, steps, hidden]
    assert tokens.shape == (batch, num_steps)
    assert features.shape == (batch, num_steps, hidden)

def test_eagle_draft_loop_with_cache(eagle_head):
    num_steps = 3
    model = EagleDraftLoop(eagle_head, num_steps=num_steps, use_cache=True)
    
    batch = 1
    hidden = 32
    head_dim = 8
    input_ids = torch.randint(0, 100, (batch, 1))
    history_features = torch.randn(batch, 1, hidden)
    past_key = torch.zeros(batch, 4, 1, head_dim)
    past_value = torch.zeros(batch, 4, 1, head_dim)
    
    tokens, features, pk, pv = model(input_ids, history_features, past_key, past_value)
    
    assert tokens.shape == (batch, num_steps)
    assert features.shape == (batch, num_steps, hidden)
    # Cache should grow by num_steps. Original 1 + 3 steps = 4
    assert pk.shape[2] == 4 

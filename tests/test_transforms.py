import pytest
import torch
from slime_parity.transforms import ParameterTransform
from training_parity.common import EvidenceError


def test_grouped_qkv_mapping_preserves_head_order():
    shapes={"q":[4,2],"k":[2,2],"v":[2,2]}
    expr={"qkv":{"op":"reshape","shape":[8,2],"value":{"op":"concat","axis":1,"values":[
        {"op":"reshape","shape":[2,2,2],"value":"q"},
        {"op":"reshape","shape":[2,1,2],"value":"k"},
        {"op":"reshape","shape":[2,1,2],"value":"v"}]}}}
    transform=ParameterTransform(shapes,expr)
    tensors={"q":torch.arange(8).reshape(4,2),"k":torch.arange(4).reshape(2,2)+20,"v":torch.arange(4).reshape(2,2)+40}
    got=transform.apply(tensors)["qkv"]
    expected=torch.stack([tensors['q'][0],tensors['q'][1],tensors['k'][0],tensors['v'][0],
                          tensors['q'][2],tensors['q'][3],tensors['k'][1],tensors['v'][1]])
    torch.testing.assert_close(got,expected)


@pytest.mark.parametrize("expr", [{"a":"q"}, {"a":"q","b":"q"}, {"a":"unknown"},
                                  {"a":{"op":"reshape","value":"q","shape":[99]}}])
def test_lossy_or_duplicated_reference_mapping_rejected(expr):
    with pytest.raises((EvidenceError,ValueError)):
        ParameterTransform({"q":[2,3],"k":[1,3]},expr)


def test_transpose_mapping_applies_to_gradients_as_well():
    mapping=ParameterTransform({"linear.weight":[2,3]}, {"native.weight":{"op":"permute","value":"linear.weight","axes":[1,0]}})
    weight=torch.arange(6,dtype=torch.float64).reshape(2,3).requires_grad_()
    (weight.square().sum()).backward()
    torch.testing.assert_close(mapping.apply({"linear.weight":weight.grad})['native.weight'],(2*weight).T)

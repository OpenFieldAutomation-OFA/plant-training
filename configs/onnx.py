onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=17,
    save_file='model.onnx',
    input_names=['input'],
    output_names=['output'],
    input_shape=None
)
codebase_config = dict(type='mmpretrain', task='Classification')
backend_config = dict(type='onnxruntime')
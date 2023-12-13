import onnxruntime

all_faces = None
selected_face_index = None
log_level = 'error'
cpu_cores = None
gpu_threads = None
gpu_vendor = None
providers = onnxruntime.get_available_providers()

if 'TensorrtExecutionProvider' in providers:
    providers.remove('TensorrtExecutionProvider')

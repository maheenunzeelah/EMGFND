
import base64
import io
import torch


def tensor_to_base64(tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def base64_to_tensor(b64_str):
    byte_data = base64.b64decode(b64_str)
    buffer = io.BytesIO(byte_data)
    return torch.load(buffer)
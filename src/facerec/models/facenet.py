"""Gerador de embeddings com o FaceNet (InceptionResnetV1) do facenet-pytorch.

Os pesos pré-treinados (VGGFace2) são baixados na primeira execução (cerca de 107 MB) e
guardados no cache do PyTorch do usuário; não são versionados no Git.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


class FaceNetEmbedder:
    dim = 512
    input_size = 160

    def __init__(self, pretrained: str = "vggface2", device: str = "cpu"):
        import torch
        from facenet_pytorch import InceptionResnetV1  # importação adiada

        self._torch = torch
        self.device = device
        self.net = InceptionResnetV1(pretrained=pretrained).eval().to(device)

    def embed(self, faces: list[Image.Image]) -> np.ndarray:
        """Embeddings L2-normalizados, com shape (n, 512), de recortes de face já alinhados."""
        torch = self._torch
        batch = np.stack(
            [
                np.asarray(f.convert("RGB").resize((self.input_size, self.input_size)), dtype=np.float32)
                for f in faces
            ]
        )
        # mesma padronização do treinamento do modelo: (x - 127.5) / 128
        x = torch.from_numpy((batch - 127.5) / 128.0).permute(0, 3, 1, 2).contiguous().to(self.device)
        with torch.no_grad():
            out = self.net(x)
            out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out.cpu().numpy().astype(np.float32)

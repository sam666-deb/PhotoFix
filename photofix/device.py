"""Pick the best available PyTorch device (Apple Silicon GPU via MPS on this Mac)."""


def get_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    print(get_device())

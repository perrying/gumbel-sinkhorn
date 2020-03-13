import numpy as np
import torch

def np_divide_image(image: np.ndarray, num_pieces: int):
    height, width = image.shape[-2:]
    piece_height = height // num_pieces
    piece_width  = width // num_pieces
    pieces = []
    for p_h in range(num_pieces):
        for p_w in range(num_pieces):
            left   = p_w * piece_width
            right  = left + piece_width
            top    = p_h * piece_height
            bottom = top + piece_height
            piece  = image[:,top:bottom,left:right]
            pieces.append(piece)
    permute_index = np.random.permutation(num_pieces**2)
    pieces = np.stack(pieces, 0) # (num_pieces, channels, height//num_pieces, width//num_pieces)
    random_pieces = pieces[permute_index]
    return (pieces, random_pieces, permute_index)

def tch_divide_image(image: torch.Tensor, num_pieces: int):
    height, width = image.shape[-2:]
    piece_height = height // num_pieces
    piece_width  = width // num_pieces
    pieces = []
    for p_h in range(num_pieces):
        for p_w in range(num_pieces):
            left   = p_w * piece_width
            right  = left + piece_width
            top    = p_h * piece_height
            bottom = top + piece_height
            piece  = image[:,top:bottom,left:right]
            pieces.append(piece)
    permute_index = torch.randperm(num_pieces**2)
    pieces = torch.stack(pieces, 0) # (num_pieces, channels, height//num_pieces, width//num_pieces)
    random_pieces = pieces[permute_index]
    return (pieces, random_pieces, permute_index)

def batch_tch_divide_image(images: torch.Tensor, num_pieces: int):
    batch_pieces, batch_random_pieces, batch_permute_index = [], [], []
    for image in images:
        pieces, random_pieces, permute_index = tch_divide_image(image, num_pieces)
        batch_pieces.append(pieces); batch_random_pieces.append(random_pieces); batch_permute_index.append(permute_index)
    return torch.stack(batch_pieces, 0), torch.stack(batch_random_pieces, 0), torch.stack(batch_permute_index, 0)

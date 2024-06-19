import torch

def find_subtensor_position(sub_tensor: torch.Tensor, main_tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Find all start indices of sub_tensor in each sequence of the main_tensor batch.
    Input:
    - sub_tensor: Tensor of shape (len_sub)
    - main_tensor: Tensor of shape (batch, len_main) or (len_main)
    Output:
    - start_indices_list: list of Tensor, each of shape (num_occurrences)
    """
    len_sub = sub_tensor.size(0)

    if len(main_tensor.size()) == 1:
        main_tensor = main_tensor.unsqueeze(0)
    batch_size = main_tensor.size(0)
    
    # List to store tensors of the start indices for each sequence in the batch
    start_indices_list = []

    # Iterate through each sequence in the batch of main_tensor
    for seq in main_tensor:
        start_indices = []
        len_seq = seq.size(0)
        # Iterate through the current sequence to find matches
        for i in range(len_seq - len_sub + 1):
            # Check if the current slice of the sequence matches sub_tensor
            if torch.equal(seq[i:i+len_sub], sub_tensor):
                start_indices.append(i)
        # Convert the list of start indices to a tensor and add it to the list
        start_indices_list.append(torch.tensor(start_indices))

    return start_indices_list
